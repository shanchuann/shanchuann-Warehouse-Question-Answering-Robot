import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = '你的openai api key'

# 自定义文本加载器，适配文件编码问题
class CustomTextLoader(TextLoader) :
    def __init__(self, file_path: str, encoding: str = "utf-8") :
        super().__init__(file_path, encoding=encoding)

    def load(self) :
        # 在这里处理文本加载，并确保编码问题
        try :
            return super().load()
        except UnicodeDecodeError :
            # 如果 utf-8 编码失败，尝试使用 gbk 编码
            return super().__init__(self.file_path, encoding="gbk").load()


# 用于处理文档加载、文本分割和向量存储
class ChatbotWithRetrieval :
    def __init__(self, data_folder: str) :
        self.data_folder = data_folder
        self.embeddings = OpenAIEmbeddings()
        self.documents = self.load_documents()
        self.vectorstore = self.create_vectorstore()
        self.qa_chain = self.create_qa_chain()

    def load_documents(self) :
        documents = []
        # 遍历文件夹加载所有 .md 文件
        for root, dirs, files in os.walk(self.data_folder) :
            for file in files :
                if file.endswith(".md") :
                    file_path = os.path.join(root, file)
                    # 使用自定义加载器，支持不同编码
                    loader = CustomTextLoader(file_path, encoding="utf-8")
                    documents.extend(loader.load())
        return documents

    def create_vectorstore(self) :
        # 使用 FAISS 向量存储
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        all_splits = text_splitter.split_documents(self.documents)
        return FAISS.from_documents(all_splits, self.embeddings)

    def create_qa_chain(self) :
        # 创建问答链
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k" : 1})
        return RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            chain_type="stuff",
            retriever=retriever
        )

    def query(self, question: str) :
        # 返回回答
        return self.qa_chain.run(question)


# Streamlit Web UI
def main() :
    st.title("仓库问答机器人")

    # 检查 session_state 是否已有 bot
    if 'bot' not in st.session_state :
        st.session_state.bot = ChatbotWithRetrieval("assets")

    question = st.text_input("请输入您的问题：")

    if question :
        response = st.session_state.bot.query(question)
        st.write("回答：", response)


if __name__ == "__main__" :
    main()