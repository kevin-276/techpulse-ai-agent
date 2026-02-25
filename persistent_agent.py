import os
import aiohttp 
import asyncio
import aiosqlite 
from typing import Annotated, Literal, TypedDict, List

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver 
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document # 👈 新增：用于构造存入数据库的文档格式

# ==========================================
# 1. Prompt (保持不变)
# ==========================================
SYSTEM_PROMPT = """你是一个具备【自我反思(Self-Reflection)】能力的 GitHub 专家 Agent。
你拥有两个工具：
1. search_local_memory（本地知识库）
2. search_github（联网搜索）

【你的标准工作流】：
步骤 1: 永远先调用 search_local_memory 检索本地知识。
步骤 2: 【关键决策】作为裁判，审视本地工具返回的结果。利用你的常识判断：返回的项目和用户想找的项目是同一个东西吗？
步骤 3: 如果你认为本地结果是“答非所问”（例如用户找 OpenClaw，却返回了 AutoGPT），说明本地没有该知识。此时，你**必须主动、静默地**调用 search_github 工具联网查询。绝对不要把错误的本地结果告诉用户，也不要向用户提问。
步骤 4: 基于正确的数据（本地的或联网查到的），向用户输出最终回答。
"""

# ==========================================
# 2. 异步 Agent 构造器 (核心：自动学习机制)
# ==========================================
def build_graph(api_key: str):
    if not api_key: raise ValueError("API Key is missing")

    # 1. 初始化 Embedding 和 向量数据库
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key 
    )

    local_vector_store = Chroma(
        persist_directory="chroma_db", 
        embedding_function=embeddings
    )

    # 🔧 工具 A：检索本地记忆 (Agentic 评判版)
    @tool
    async def search_local_memory(query: str):
        """优先使用此工具！在本地向量数据库中检索项目信息。"""
        print(f"--- [Backend] 🧠 正在检索本地知识库: {query} ---")
        try:
            # 💡 核心升级：同时获取相似度得分 (Score)
            # 在 Chroma 中，默认使用 L2 距离，数值越小代表越相似
            results = await local_vector_store.asimilarity_search_with_score(query, k=2)
            
            if not results:
                return "本地知识库为空，请立刻调用 search_github 工具联网查询。"
            
            info_list = []
            for doc, score in results:
                repo_name = doc.metadata.get('repo', 'Unknown')
                # 把分数也喂给大模型，辅助它做决策
                info_list.append(f"项目: {repo_name}\n描述: {doc.page_content}\n向量距离得分(越接近0越匹配): {score:.2f}")
            
            # 💡 核心升级：在工具返回值中，植入“反思指令”
            return (
                "【本地检索结果】如下：\n"
                "⚠️ 请你作为裁判，评估以下结果是否真的符合用户的查询意图。\n"
                "如果距离得分过大，或者项目描述明显不符，请忽略此信息，并立刻调用 search_github 工具！\n\n"
                + "\n---\n".join(info_list)
            )
            
        except Exception as e:
            return f"本地检索发生错误: {e}，请改用 search_github。"
    # 🔧 工具 B：联网搜 GitHub 并自动学习 (读 + 写)
    @tool
    async def search_github(query: str):
        """当本地记忆找不到时，使用此工具搜索 GitHub 并自动学习新知识。"""
        print(f"--- [Backend] 🌐 正在启动联网搜索: {query} ---")
        url = f"https://api.github.com/search/repositories?q={query}"
        # 从环境变量读取 Token，如果没配置，就提供一个空字符串防止报错
        # 🌟 修复点：更安全地读取和拼装 Headers
        github_token = os.environ.get("GITHUB_TOKEN", "").strip()
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
            
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200: return f"Error: Status {response.status}"
                    data = await response.json()
                    if 'items' not in data: return f"Error: {data}"
                    
                    results = []
                    docs_to_learn = [] # 用于准备存入数据库的列表
                    
                    for item in data['items'][:3]: # 取前 3 个高质量结果
                        repo_name = item['full_name']
                        stars = item['stargazers_count']
                        desc = item['description']
                        
                        # 构造纯文本信息给大模型看
                        content_str = f"Name: {repo_name}, Stars: {stars}, Desc: {desc}"
                        results.append(content_str)
                        
                        # 构造 Document 对象给向量数据库吃
                        # 把名字和描述拼在一起作为“语义内容”，把仓库名作为“元数据”
                        doc = Document(
                            page_content=f"{repo_name} 是一个 GitHub 项目。描述：{desc}",
                            metadata={"repo": repo_name}
                        )
                        docs_to_learn.append(doc)
                    
                    # 🌟 核心动作：自动学习！将新知识写入 ChromaDB
                    if docs_to_learn:
                        print(f"--- [Backend] 💾 正在自动学习！将 {len(docs_to_learn)} 个新项目写入本地知识库 ---")
                        # 异步添加文档到向量库
                        await local_vector_store.aadd_documents(docs_to_learn)
                    
                    return "\n".join(results)
        except Exception as e:
            return f"Search Network Error: {e}"

    # --- 后续组装图逻辑保持不变 ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key, temperature=0)
    tools = [search_local_memory, search_github]
    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def agent_node(state: AgentState):
        messages = state['messages']
        sys_msg = SystemMessage(content=SYSTEM_PROMPT)
        response = llm_with_tools.invoke([sys_msg] + messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def should_continue(state: AgentState) -> Literal["tools", END]:
        if state['messages'][-1].tool_calls: return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow

# ==========================================
# 3. 辅助功能 (保持不变)
# ==========================================
DB_PATH = "agent_memory.sqlite"

def clear_memory_sync(thread_id: str):
    import sqlite3
    # 如果数据库文件都不存在，说明本来就是空的，直接返回成功
    if not os.path.exists(DB_PATH): 
        return True 
        
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # 兼容 LangGraph 新老版本的不同表名
            tables_to_clear = ["checkpoints", "checkpoint_blobs", "checkpoint_writes", "writes"]
            
            for table in tables_to_clear:
                try:
                    cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
                except sqlite3.OperationalError:
                    # 如果某个表不存在，忽略报错继续删下一个
                    pass 
            conn.commit()
        return True
    except Exception as e:
        print(f"Error clearing memory: {e}")
        return False

def get_existing_users_sync() -> List[str]:
    import sqlite3
    if not os.path.exists(DB_PATH): return []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            rows = cursor.fetchall()
            return [row[0] for row in rows]
    except Exception as e:
        print(f"Error reading users: {e}")
        return []