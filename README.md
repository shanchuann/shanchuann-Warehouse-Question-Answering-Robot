# VChart智能问答机器人

## 项目简介

本项目是一个基于 LangChain、FAISS 和 OpenAI API 的 仓库问答机器人，用于从仓库文档中检索相关信息并回答用户问题。该机器人采用 Streamlit 作为前端，结合文本嵌入、向量数据库和大语言模型，实现智能问答功能。

## 功能特点

文档加载：支持从本地文件夹中加载文本数据。

文本分割：采用 RecursiveCharacterTextSplitter 进行文本切片，以便高效索引和检索。

向量存储：使用 FAISS 作为向量数据库，存储文档的嵌入向量。

问答检索：基于 OpenAI API 进行自然语言处理，并结合 FAISS 进行相似度搜索，提供精准回答。

可视化界面：使用 Streamlit 构建交互界面，用户可直接在网页端输入问题并获取答案。

## 安装及环境配置

### 依赖环境

请确保已安装 Python（建议使用 Python 3.8 及以上），并安装以下依赖：

`pip install streamlit langchain faiss-cpu openai langchain_openai langchain_community`

### 配置 OpenAI API 密钥

本项目使用 OpenAI API 进行嵌入和问答，请在环境变量中配置 API Key：

`export OPENAI_API_KEY="your-api-key"`

或在 Windows 命令行（CMD / PowerShell）中设置：

`$env:OPENAI_API_KEY="your-api-key"` 

## 代码结构

Warehouse-Question-Answering-Robot/ 
```
│── assets/ # 存放需要加载的文本文件 
│── chatbot.py # 主要的 Streamlit 应用代码 
│── .venv/ # 虚拟环境（可选）
│── README.md # 项目说明文档
```

## 运行方式

确保已安装所有依赖并正确配置 OpenAI API Key 后，在终端或命令行运行以下命令启动 Streamlit 应用：

`streamlit run chatbot.py`

## 常见问题及解决方案

1. 运行 streamlit run chatbot.py 之后只有标题，没有其他内容

    可能原因：
    
    后端代码异常：检查 chatbot.py 是否报错，尤其是 API 调用或向量数据库加载部分。
    
    API 超时：如果 OpenAI API 请求超时，Streamlit 可能无法正确渲染内容。
    
    向量数据库未正确加载：FAISS 可能未正确初始化或数据未存入数据库。
    
    解决方案：
    
    在 chatbot.py 中增加日志打印，检查 API 调用是否成功： import logging logging.basicConfig(level=logging.INFO) 
    
    测试 OpenAI API 是否可用： from openai import OpenAI client = OpenAI() response = client.embeddings.create(input="测试", model="text-embedding-ada-002") print(response) 
    
    确保 faiss-cpu 已正确安装： pip install faiss-cpu 

2. ModuleNotFoundError: No module named 'langchain.text_splitters'

    可能原因：
    
    LangChain 在新版本中已将 text_splitters 移动到 langchain_community。
    
    解决方案：
    
    修改 chatbot.py 中的导入代码： from langchain_community.text_splitters import RecursiveCharacterTextSplitter 

3. openai.OpenAIError: The api_key client option must be set

    可能原因：
    
    未正确设置 OpenAI API Key。
    
    解决方案：
    
    检查环境变量是否正确：
     
    `echo $OPENAI_API_KEY # Linux/macOS `

    `echo %OPENAI_API_KEY% # Windows CMD `

    `$env:OPENAI_API_KEY # PowerShell `
    
    在代码中直接设置 API Key：

    ```
   import os os.environ["OPENAI_API_KEY"] = "your-api-key" 
    ```
4. APITimeoutError: Request timed out
    
    可能原因：
    
    网络问题：连接 OpenAI 服务器超时。
    
    API 服务器负载过高：OpenAI 可能正在维护或请求量过大。
    
    解决方案：
    
    检查网络连接，尝试使用 VPN 访问 OpenAI API。
    
    增加 API 请求的超时时间： from openai import OpenAI client = OpenAI(timeout=60) # 设置超时时间为 60 秒 
    
    访问 OpenAI API 状态页面 检查服务器是否正常运行。