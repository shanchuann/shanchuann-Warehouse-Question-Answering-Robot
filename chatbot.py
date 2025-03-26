import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import requests
import time
from datetime import datetime
from typing import Dict, List

# 语言配置
TRANSLATIONS = {
    "zh": {
        "title": "仓库问答机器人",
        "new_chat": "新对话",
        "clear_chat": "清空对话",
        "export_chat": "导出对话",
        "reinit": "重新初始化",
        "update_kb": "更新知识库",
        "input_placeholder": "在这里输入您的问题...",
        "submit_button": "提交问题",
        "stop_button": "停止",
        "welcome_title": "你好！我是仓库问答助手",
        "welcome_text": "请在下方输入您的问题，我会尽力帮您解答",
        "user_title": "用户",
        "assistant_title": "助手",
        "no_export": "当前没有对话可导出",
        "export_filename": "对话记录",
        "input_required": "请输入问题",
        "thinking": "正在思考...",
        "features_title": "功能按钮说明",
        "features": {
            "new_chat": "开始一个全新的对话",
            "clear_chat": "清除当前对话记录",
            "export_chat": "保存对话记录到文件",
            "reinit": "重新加载文档和模型",
            "update_kb": "更新知识库内容"
        },
        "question_types_title": "支持的问题类型",
        "question_types": [
            "关于文档内容的查询",
            "文档相关的解释说明",
            "具体功能的使用方法"
        ],
        "notes_title": "注意事项",
        "notes": [
            "问题要尽量具体",
            "如果回答不准确，可以换个方式提问",
            "支持中文提问"
        ],
        "system_info_title": " 系统信息",
        "docs_loaded": "已加载文档数",
        "chat_id": "当前对话ID",
        "system_status": "系统状态",
        "initializing": "正在初始化..."
    },
    "en": {
        "title": "Repository Q&A Bot",
        "new_chat": "New Chat",
        "clear_chat": "Clear Chat",
        "export_chat": "Export Chat",
        "reinit": "Reinitialize",
        "update_kb": "Update Knowledge Base",
        "input_placeholder": "Type your question here...",
        "submit_button": "Submit",
        "stop_button": "Stop",
        "welcome_title": "Hi! I'm your Repository Assistant",
        "welcome_text": "Please enter your question below, I'll do my best to help",
        "user_title": "User",
        "assistant_title": "Assistant",
        "no_export": "No conversation to export",
        "export_filename": "chat_history",
        "input_required": "Please enter a question",
        "thinking": "Thinking...",
        "features_title": "Features",
        "features": {
            "new_chat": "Start a new conversation",
            "clear_chat": "Clear current chat history",
            "export_chat": "Save chat history to file",
            "reinit": "Reload documents and model",
            "update_kb": "Update knowledge base"
        },
        "question_types_title": "Supported Question Types",
        "question_types": [
            "Document content queries",
            "Documentation explanations",
            "Feature usage instructions"
        ],
        "notes_title": "Notes",
        "notes": [
            "Be specific with questions",
            "Try rephrasing if answer is unclear",
            "Supports both English and Chinese"
        ],
        "system_info_title": "System Info",
        "docs_loaded": "Documents Loaded",
        "chat_id": "Chat ID",
        "system_status": "System Status",
        "initializing": "Initializing..."
    }
}

def get_text(key: str, lang: str) -> str:
    """获取指定语言的文本"""
    return TRANSLATIONS[lang][key]

# 设置页面配置
st.set_page_config(
    page_title="仓库问答机器人",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 整体页面样式 */
    .main .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background-color: #2C3E50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        border: none;
        margin: 0.25rem 0;
        transition: all 0.2s ease;
        font-size: 0.875rem;
    }
    .stButton > button:hover {
        background-color: #34495E;
        transform: translateY(-1px);
    }
    
    /* 停止按钮样式 */
    .stop-button > button {
        background-color: #E74C3C;
    }
    .stop-button > button:hover {
        background-color: #C0392B;
    }
    
    /* 消息气泡样式 */
    .user-question {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        position: relative;
        max-width: 90%;
        margin-left: auto;
    }
    
    .bot-answer {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        position: relative;
        max-width: 90%;
        border: 1px solid #E2E8F0;
    }
    
    /* 消息头部样式 */
    .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
        color: #4A5568;
    }
    
    .message-content {
        font-size: 1rem;
        line-height: 1.5;
        color: #2D3748;
    }
    
    /* 欢迎消息样式 */
    .welcome-message {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 1rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .welcome-message h3 {
        color: #2D3748;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
    
    .welcome-message p {
        color: #4A5568;
        font-size: 1.1rem;
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        background-color: #FFFFFF;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3182CE;
        box-shadow: 0 0 0 3px rgba(49,130,206,0.1);
    }
    
    /* 系统信息框样式 */
    .status-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.875rem;
        color: #4A5568;
    }
    
    /* 工具栏样式 */
    .stButton.toolbar > button {
        background-color: #F7FAFC;
        color: #2D3748;
        border: 1px solid #E2E8F0;
        font-size: 0.875rem;
    }
    
    .stButton.toolbar > button:hover {
        background-color: #EDF2F7;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background-color: #F8F9FA;
    }
    
    /* 展开器样式 */
    .streamlit-expanderHeader {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        color: #2D3748;
    }
    
    /* 输入区域容器样式 */
    .stTextInput, .stButton {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* 输入框样式 */
    .stTextInput > div {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .stTextInput > div > div {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .stTextInput > div > div > input {
        margin: 0 !important;
        padding: 0.5rem 1rem !important;
        height: 38px !important;
        min-height: 38px !important;
        line-height: 1.5 !important;
        box-sizing: border-box !important;
    }
    
    /* 按钮样式调整 */
    .stButton > button {
        margin: 0 !important;
        height: 38px !important;
        min-height: 38px !important;
        padding: 0.5rem 1rem !important;
        line-height: 1.5 !important;
        box-sizing: border-box !important;
    }
    
    /* 底部固定区域样式 */
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        background-color: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# 自定义文本加载器，适配文件编码问题
class CustomTextLoader(TextLoader):
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        super().__init__(file_path, encoding=encoding)

    def load(self):
        try:
            return super().load()
        except UnicodeDecodeError:
            return super().__init__(self.file_path, encoding="gbk").load()

def check_ollama_service():
    """检查 Ollama 服务是否可用"""
    try:
        # 尝试多个端点
        endpoints = [
            "http://127.0.0.1:11434",
            "http://localhost:11434",
            "http://[::1]:11434"
        ]
        
        st.sidebar.write("正在检查 Ollama 服务...")
        
        for endpoint in endpoints:
            try:
                st.sidebar.write(f"尝试连接: {endpoint}")
                # 先尝试基础连接
                response = requests.get(f"{endpoint}/", timeout=5)
                st.sidebar.write(f"基础连接响应: {response.status_code}")
                
                # 再尝试 API 端点
                api_response = requests.post(
                    f"{endpoint}/api/embeddings",
                    json={"model": "llama2", "prompt": "test"},
                    timeout=5
                )
                st.sidebar.write(f"API 响应: {api_response.status_code}")
                
                if api_response.status_code == 200:
                    st.sidebar.success(f"成功连接到 {endpoint}")
                    return True
            except Exception as e:
                st.sidebar.warning(f"连接 {endpoint} 失败: {str(e)}")
                continue
                
        st.sidebar.error("所有连接尝试均失败")
        return False
    except Exception as e:
        st.sidebar.error(f"检查服务时出错：{str(e)}")
        return False

def check_model_available(model_name="llama2"):
    """检查模型是否已下载"""
    try:
        endpoints = [
            "http://127.0.0.1:11434",
            "http://localhost:11434",
            "http://[::1]:11434"
        ]
        
        st.sidebar.write("正在检查模型状态...")
        
        for endpoint in endpoints:
            try:
                # 尝试直接使用模型
                response = requests.post(
                    f"{endpoint}/api/embeddings",
                    json={"model": model_name, "prompt": "test"},
                    timeout=5
                )
                
                if response.status_code == 200:
                    st.sidebar.success(f"模型 {model_name} 可用")
                    return True
                elif response.status_code == 404:
                    st.sidebar.warning(f"模型 {model_name} 未找到")
                else:
                    st.sidebar.warning(f"检查模型时收到意外响应: {response.status_code}")
                    
            except Exception as e:
                st.sidebar.warning(f"检查模型时出错: {str(e)}")
                continue
                
        return False
    except Exception as e:
        st.sidebar.error(f"检查模型时出错：{str(e)}")
        return False

# 用于处理文档加载、文本分割和向量存储
class ChatbotWithRetrieval:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.documents = None
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = None
        self.vector_store_path = "vector_store"  # 向量存储保存路径
        
        # 尝试加载现有的向量存储
        if self.load_existing_vectorstore():
            st.sidebar.success("已加载现有向量存储")
        else:
            self.initialize_bot()

    def load_existing_vectorstore(self):
        """尝试加载现有的向量存储"""
        try:
            if os.path.exists(self.vector_store_path):
                # 配置 embeddings
                base_url = "http://127.0.0.1:11434"
                self.embeddings = OllamaEmbeddings(
                    model="llama2",
                    base_url=base_url
                )
                
                # 加载向量存储，允许反序列化
                self.vectorstore = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # 创建问答链
                self.qa_chain = self.create_qa_chain(base_url)
                return True
            return False
        except Exception as e:
            st.sidebar.warning(f"加载现有向量存储失败：{str(e)}")
            return False

    def save_vectorstore(self):
        """保存向量存储到本地"""
        try:
            if self.vectorstore:
                self.vectorstore.save_local(self.vector_store_path)
                st.sidebar.success("向量存储已保存到本地")
        except Exception as e:
            st.sidebar.error(f"保存向量存储失败：{str(e)}")

    def initialize_bot(self):
        """初始化机器人的所有组件"""
        try:
            # 检查 Ollama 服务
            if not check_ollama_service():
                st.error("Ollama 服务未运行！")
                st.info("""
                ### 请按以下步骤操作：
                1. 打开新的命令行窗口（以管理员身份运行）
                2. 运行以下命令停止现有服务：
                   ```
                   taskkill /F /IM ollama.exe
                   ```
                3. 等待几秒后重新启动服务：
                   ```
                   ollama serve
                   ```
                4. 在新窗口中测试服务：
                   ```
                   curl http://127.0.0.1:11434/
                   ```
                5. 如果测试成功，刷新此页面
                
                如果还是不行：
                1. 检查任务管理器中是否有多个 ollama 进程
                2. 检查端口 11434 是否被占用
                3. 尝试重启电脑后重试
                """)
                return
            
            if not check_model_available("llama2"):
                st.error("llama2 模型未找到！")
                st.info("""
                ### 请按以下步骤操作：
                1. 确保 Ollama 服务正在运行
                2. 在新的命令行窗口中运行：
                   ```
                   ollama pull llama2
                   ```
                3. 等待下载完成（保持窗口开着）
                4. 下载完成后运行：
                   ```
                   ollama list
                   ```
                5. 确认看到 llama2 后刷新此页面
                """)
                return

            st.success("Ollama 服务正常运行")
            st.info("正在初始化机器人...")
            
            # 配置 embeddings
            base_url = "http://127.0.0.1:11434"
            self.embeddings = OllamaEmbeddings(
                model="llama2",
                base_url=base_url
            )
            
            st.info("正在加载文档...")
            self.documents = self.load_documents()
            if not self.documents:
                st.error("未找到任何文档！")
                return
            st.success(f"已加载 {len(self.documents)} 个文档")
            
            st.info("正在创建向量存储...")
            self.vectorstore = self.create_vectorstore()
            
            # 保存向量存储到本地
            self.save_vectorstore()
            
            st.info("正在初始化问答链...")
            self.qa_chain = self.create_qa_chain(base_url)
            
        except Exception as e:
            st.error(f"初始化失败：{str(e)}")
            st.info("""
            ### 请尝试以下解决方案：
            1. 检查 Ollama 服务状态
            2. 确保网络连接正常
            3. 重启应用
            4. 查看详细错误信息
            """)
            raise e

    def load_documents(self):
        """加载文档"""
        documents = []
        processed_files = set()  # 用于跟踪已处理的文件
        
        # 指定要加载的目录
        target_dirs = [
            os.path.join(self.data_folder, "guide"),
            os.path.join(self.data_folder, "api"),
            os.path.join(self.data_folder, "faq")
        ]
        
        st.sidebar.write("🔍 开始搜索文档...")
        
        # 只处理指定目录中的 .md 文件
        for target_dir in target_dirs:
            if not os.path.exists(target_dir):
                st.sidebar.warning(f"目录不存在：{target_dir}")
                continue
            
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    if not file.endswith(".md"):
                        continue
                        
                    file_path = os.path.join(root, file)
                    
                    # 跳过项目自身的 README.md
                    if file.lower() == "readme.md" and "assets" in file_path.split(os.sep)[-2:]:
                        continue
                    
                    # 检查文件是否已处理
                    if file_path in processed_files:
                        st.sidebar.info(f"跳过重复文件：{file}")
                        continue
                        
                    try:
                        st.sidebar.info(f"正在加载：{file}")
                        loader = CustomTextLoader(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                        processed_files.add(file_path)
                        st.sidebar.success(f"已加载：{file}")
                    except Exception as e:
                        st.sidebar.error(f"加载失败 {file}：{str(e)}")
                        continue
        
        if not documents:
            st.error("""
            ### 未找到任何文档！
            
            请确保以下目录中包含 .md 文件：
            - assets/guide/
            - assets/api/
            - assets/faq/
            
            当前搜索路径：{}
            """.format([os.path.abspath(d) for d in target_dirs]))
            raise Exception("未找到任何支持的文档文件")
            
        st.sidebar.success(f"共加载了 {len(documents)} 个文档")
        return documents

    def create_vectorstore(self):
        """创建向量存储"""
        try:
            st.sidebar.info("正在处理文档...")
            # 文本分割
            text_splitter = CharacterTextSplitter(
                chunk_size=500,  # 减小块大小
                chunk_overlap=50,
                separator="\n\n",  # 使用双换行作为分隔符
                length_function=len,
                is_separator_regex=False
            )
            
            # 使用集合去重
            unique_docs = list({doc.page_content: doc for doc in self.documents}.values())
            st.sidebar.info(f"去重后文档数量：{len(unique_docs)}")
            
            # 分批处理文档
            batch_size = 50  # 减小批处理大小
            all_splits = []
            
            for i in range(0, len(unique_docs), batch_size):
                batch = unique_docs[i:i + batch_size]
                splits = text_splitter.split_documents(batch)
                all_splits.extend(splits)
                st.sidebar.info(f"已处理 {min(i + batch_size, len(unique_docs))}/{len(unique_docs)} 个文档")
            
            st.sidebar.success(f"文档已分割为 {len(all_splits)} 个片段")
            
            # 创建向量存储
            st.sidebar.info("正在创建向量索引...")
            vectorstore = FAISS.from_documents(all_splits, self.embeddings)
            st.sidebar.success("向量索引创建完成")
            
            return vectorstore
        except Exception as e:
            st.sidebar.error(f"创建向量存储失败：{str(e)}")
            raise e

    def create_qa_chain(self, base_url):
        """创建问答链"""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 1}
        )
        return RetrievalQA.from_chain_type(
            llm=OllamaLLM(
                model="llama2",
                base_url=base_url
            ),
            chain_type="stuff",
            retriever=retriever
        )

    def query(self, question: str):
        """处理用户查询"""
        try:
            if not self.qa_chain:
                raise Exception("问答链未初始化")
            return self.qa_chain.run(question)
        except Exception as e:
            return f"发生错误：{str(e)}"

# Streamlit Web UI
def main():
    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question" not in st.session_state:
        st.session_state.question = ""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = 1
    if "language" not in st.session_state:
        st.session_state.language = "zh"

    # 添加语言切换
    lang = st.sidebar.radio(
        "🌍 Language / 语言",
        options=["中文", "English"],
        index=0 if st.session_state.language == "zh" else 1,
        key="lang_select",
        horizontal=True
    )
    st.session_state.language = "zh" if lang == "中文" else "en"
    
    # 添加分隔线
    st.sidebar.markdown("---")
    
    # 获取当前语言
    current_lang = st.session_state.language

    # 设置标题
    st.title(get_text("title", current_lang))

    # 创建两列布局
    col1, col2 = st.columns([3, 1])

    with col1:
        # 创建顶部工具栏
        with st.container():
            tool_col1, tool_col2, tool_col3, tool_col4, tool_col5 = st.columns(5)
            
            with tool_col1:
                if st.button("🔄 " + get_text("new_chat", current_lang), use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.conversation_id += 1
                    st.rerun()
                    
            with tool_col2:
                if st.button("🗑️ " + get_text("clear_chat", current_lang), use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
                    
            with tool_col3:
                if st.button("💾 " + get_text("export_chat", current_lang), use_container_width=True):
                    if st.session_state.messages:
                        conversation_text = "\n\n".join(st.session_state.messages)
                        st.download_button(
                            "📥 Save",
                            conversation_text,
                            file_name=f"{get_text('export_filename', current_lang)}_{st.session_state.conversation_id}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning(get_text("no_export", current_lang))
                        
            with tool_col4:
                if st.button("⚡ " + get_text("reinit", current_lang), use_container_width=True):
                    if os.path.exists("vector_store"):
                        import shutil
                        shutil.rmtree("vector_store")
                    if 'bot' in st.session_state:
                        del st.session_state.bot
                    st.rerun()
                    
            with tool_col5:
                if st.button("📚 " + get_text("update_kb", current_lang), use_container_width=True):
                    if os.path.exists("vector_store"):
                        import shutil
                        shutil.rmtree("vector_store")
                    if 'bot' in st.session_state:
                        del st.session_state.bot
                    st.session_state.bot = ChatbotWithRetrieval("assets")
                    st.rerun()

        # 创建聊天界面
        chat_container = st.container()
        
        # 显示对话历史
        with chat_container:
            if st.session_state.messages:
                for message in st.session_state.messages:
                    if message.startswith("问：") or message.startswith("Q:"):
                        st.markdown(f'''
                        <div class="user-question">
                            <div class="message-header">
                                <div>👤 {get_text("user_title", current_lang)}</div>
                                <div class="timestamp">{datetime.now().strftime("%H:%M")}</div>
                            </div>
                            <div class="message-content">{message[2:]}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="bot-answer">
                            <div class="message-header">
                                <div>🤖 {get_text("assistant_title", current_lang)}</div>
                                <div class="timestamp">{datetime.now().strftime("%H:%M")}</div>
                            </div>
                            <div class="message-content">{message[2:]}</div>
                        </div>
                        ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="welcome-message">
                    <h3>👋 {get_text("welcome_title", current_lang)}</h3>
                    <p>{get_text("welcome_text", current_lang)}</p>
                </div>
                ''', unsafe_allow_html=True)

        # 添加输入区域
        with st.container():
            st.markdown("""
            <div style="position: fixed; bottom: 0; left: 50%; transform: translateX(-50%); width: 100%; max-width: 800px; background-color: white; padding: 1rem; box-shadow: 0 -2px 10px rgba(0,0,0,0.1); z-index: 1000;">
            </div>
            """, unsafe_allow_html=True)
            
            input_col, submit_col, stop_col = st.columns([6, 1, 1])
            with input_col:
                question = st.text_input(
                    "问题输入",  # 添加标签
                    key="question_input",
                    placeholder=get_text("input_placeholder", current_lang),
                    label_visibility="collapsed"
                )

            with submit_col:
                submit = st.button(
                    "🚀 提问",
                    key="submit_button",
                    use_container_width=True,
                    type="primary"
                )

            with stop_col:
                stop = st.button(
                    "⏹️ 停止",
                    key="stop_button",
                    use_container_width=True,
                    type="secondary"
                )

    with col2:
        # 添加功能说明
        with st.expander(get_text("features_title", current_lang), expanded=True):
            for key, value in get_text("features", current_lang).items():
                st.markdown(f"- **{get_text(key, current_lang)}**: {value}")

        # 添加问题类型说明
        with st.expander(get_text("question_types_title", current_lang), expanded=True):
            for qtype in get_text("question_types", current_lang):
                st.markdown(f"- {qtype}")

        # 添加注意事项
        with st.expander(get_text("notes_title", current_lang), expanded=True):
            for note in get_text("notes", current_lang):
                st.markdown(f"- {note}")

        # 添加系统信息
        with st.expander(get_text("system_info_title", current_lang), expanded=True):
            if hasattr(st.session_state, 'bot'):
                docs_count = len(st.session_state.bot.documents) if st.session_state.bot.documents is not None else 0
                st.markdown(f"""
                <div class="status-box">
                    <p>{get_text("docs_loaded", current_lang)}: {docs_count}</p>
                    <p>{get_text("chat_id", current_lang)}: {st.session_state.conversation_id}</p>
                    <p>{get_text("system_status", current_lang)}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-box">
                    <p>{get_text("initializing", current_lang)}</p>
                </div>
                """, unsafe_allow_html=True)

    # 处理提交
    if submit and question:
        try:
            if 'bot' not in st.session_state:
                with st.spinner("🤖 " + get_text("initializing", current_lang)):
                    st.session_state.bot = ChatbotWithRetrieval("assets")

            if hasattr(st.session_state, 'bot'):
                with st.spinner("🤔 " + get_text("thinking", current_lang)):
                    if not stop:
                        response = st.session_state.bot.query(question)
                        prefix = "问：" if current_lang == "zh" else "Q: "
                        answer_prefix = "答：" if current_lang == "zh" else "A: "
                        st.session_state.messages.append(f"{prefix}{question}")
                        st.session_state.messages.append(f"{answer_prefix}{response}")
                        st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    elif submit:
        st.warning("⚠️ " + get_text("input_required", current_lang))

if __name__ == "__main__":
    main()