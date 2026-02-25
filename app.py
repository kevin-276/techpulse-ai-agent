import os
import streamlit as st

# ==========================================
# 0. 云端环境 SQLite 兼容性补丁
# ==========================================
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

# ==========================================
# 🌟 核心修复：从云端 Secrets 动态读取，绝不死写！
# ==========================================
# 这样写，云端会读取后台配置，本地运行时如果没配置就不会强行覆盖报错
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "TechPulse_Agent_Cloud")

    
import time
import asyncio
import aiosqlite 
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage, AIMessage

# 引入后端逻辑
from persistent_agent import build_graph, clear_memory_sync, get_existing_users_sync

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="TechPulse AI Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 侧边栏
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=60)
    st.title("控制中心 (Async)")
    st.divider()
    
    api_key = st.text_input("Gemini API Key", type="password")

    st.subheader("👤 档案切换")
    existing_users = get_existing_users_sync() 
    user_options = ["➕ 新建档案"] + existing_users
    selected_option = st.selectbox("选择当前用户", user_options, index=1 if existing_users else 0)
    
    if selected_option == "➕ 新建档案":
        user_id = st.text_input("新用户 ID", value="New_User")
    else:
        user_id = selected_option
        st.success(f"🟢 {user_id}")

    # --- 危险操作区 ---
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 刷新"): 
            st.rerun() # 重新运行整个脚本，刷新状态
    with col2:
        if st.button("🗑️ 清空"):
            # 1. 调用后端的删除逻辑
            if clear_memory_sync(user_id):
                # 2. 强行清空前端当前的对话数组
                st.session_state.messages = [] 
                st.toast(f"【{user_id}】的大脑已格式化", icon="🤯")
                time.sleep(1)
                st.rerun() # 3. 刷新页面，让清空生效
            else:
                st.error("清空记忆失败，请查看终端报错")


if not api_key:
    st.warning("👈 请输入 API Key")
    st.stop()

# ==========================================
# 3. 历史记录回填 (修复版：异步拉取)
# ==========================================
# 强制初始化为 None，确保第一次打开网页时也能触发拉取
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = None
    st.session_state.messages = []

# 检测到用户切换，或者首次加载页面
if st.session_state.current_user_id != user_id:
    st.session_state.current_user_id = user_id
    st.session_state.messages = [] 
    
    # --- 定义异步拉取器 ---
    async def fetch_history():
        async with aiosqlite.connect("agent_memory.sqlite") as conn:
            memory = AsyncSqliteSaver(conn)
            workflow = build_graph(api_key)
            app = workflow.compile(checkpointer=memory)
            config = {"configurable": {"thread_id": user_id}}
            return await app.aget_state(config) # 使用 aget_state 获取状态

    try:
        # 执行异步拉取
        state_snapshot = asyncio.run(fetch_history())
        
        # 将拉取到的历史塞回 UI 列表
        if state_snapshot.values and "messages" in state_snapshot.values:
            for msg in state_snapshot.values["messages"]:
                if isinstance(msg, HumanMessage):
                    st.session_state.messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content:
                    st.session_state.messages.append({"role": "assistant", "content": msg.content})
        
        st.toast(f"已同步 {user_id} 的历史对话", icon="📂")
    except Exception as e:
        st.error(f"加载历史记录失败: {e}")

# ==========================================
# 4. 聊天渲染
# ==========================================
st.title("🤖 TechPulse AI Agent")
st.markdown("##### 🚀 你的专属检索AI助手")
st.divider()

if not st.session_state.messages:
    st.info("👋 欢迎！请输入多个项目（如：LangChain, AutoGPT, Pandas）测试并发速度。")

for msg in st.session_state.messages:
    avatar_icon = "🤖" if msg["role"] == "assistant" else "🧐"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])

# ==========================================
# 5. 异步核心逻辑
# ==========================================
if prompt := st.chat_input("输入查询内容..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧐"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": user_id}}
    
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        # --- 定义异步运行器 ---
        async def run_async_pipeline():
            full_resp = ""
            start_time = time.time() # ⏱️ 1. 记录开始时间
            
            async with aiosqlite.connect("agent_memory.sqlite") as conn:
                memory = AsyncSqliteSaver(conn)
                workflow = build_graph(api_key)
                app = workflow.compile(checkpointer=memory)
                
                with st.status("⚡ Async Agent 并发检索中...", expanded=True) as status:
                    inputs = {"messages": [HumanMessage(content=prompt)]}
                    
                    async for event in app.astream(inputs, config=config):
                        if "agent" in event:
                            msg = event["agent"]["messages"][-1]
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    status.write(f"👉 启动任务: **{tc['name']}**")
                            else:
                                full_resp = msg.content
                                message_placeholder.markdown(full_resp)
                        
                        if "tools" in event:
                            status.write("✅ 数据流回传")
                    
                    # 兜底与状态读取
                    snapshot = await app.aget_state(config) 
                    if not full_resp and snapshot.values['messages']:
                        last_msg = snapshot.values['messages'][-1]
                        if last_msg.type == "ai":
                            full_resp = last_msg.content
                            message_placeholder.markdown(full_resp)
                    
                    status.update(label="✅ 完成", state="complete", expanded=False)
                    
                    # 🪙 2. 提取 Token 与计算耗时
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    token_info_str = ""
                    if snapshot.values['messages']:
                        final_msg = snapshot.values['messages'][-1]
                        # Langchain 会将 token 消耗存在 usage_metadata 字典中
                        if hasattr(final_msg, 'usage_metadata') and final_msg.usage_metadata:
                            in_tokens = final_msg.usage_metadata.get('input_tokens', 0)
                            out_tokens = final_msg.usage_metadata.get('output_tokens', 0)
                            token_info_str = f" | 🪙 Tokens: In {in_tokens}, Out {out_tokens}"
                    
                    # 3. 在 UI 显示监控数据
                    st.caption(f"⏱️ 耗时: {elapsed_time:.2f}s {token_info_str}")
            
            return full_resp

        # --- 驱动异步循环 ---
        try:
            full_response = asyncio.run(run_async_pipeline())
            
            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
        except Exception as e:
            st.error(f"Async Loop Error: {e}")