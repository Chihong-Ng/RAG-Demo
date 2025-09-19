# RAG-Demo: 基于LangChain的智能文档问答系统

一个基于检索增强生成（RAG）技术的智能文档问答系统，使用LangChain、DeepSeek和Streamlit构建。用户可以通过自然语言与上传的PDF文档进行交互，获取精准的答案。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://my-rag-demo.streamlit.app/) 


## 功能特点

-  **多PDF文档支持**：可同时上传和处理多个PDF文件
-  **智能检索**：基于FAISS向量数据库实现高效语义搜索
-  **AI对话**：使用DeepSeek大模型进行自然语言问答
-  **工具调用**：集成LangChain Agent实现智能工具调用
-  **友好界面**：使用Streamlit构建直观的Web界面
-  **持久化存储**：向量数据库本地持久化，无需重复处理相同文档

## 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/Chihong-Ng/RAG-Demo.git
   cd RAG-Demo
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **设置环境变量**
   
   创建 `.env` 文件并添加你的API密钥：
   ```
   DEEPSEEK_API_KEY=您的DeepSeek_API_Key
   DASHSCOPE_API_KEY=您的DashScope_API_Key
   ```

4. **运行应用**
   ```bash
   streamlit run main.py
   ```

5. **打开浏览器**
   
   访问 `http://localhost:8501` 开始使用应用
