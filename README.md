# RAG-Demo: 基于LangChain的智能文档问答系统

一个基于检索增强生成（RAG）技术的智能文档问答系统，使用LangChain、DeepSeek和Streamlit构建。用户可以通过自然语言与上传的PDF文档进行交互，获取精准的答案。

## 功能特点

- **多PDF文档支持**：可同时上传和处理多个PDF文件
- **智能检索**：基于FAISS向量数据库实现高效语义搜索
- **AI对话**：使用DeepSeek大模型进行自然语言问答
- **工具调用**：集成LangChain Tool实现智能工具调用
- **友好界面**：使用Streamlit构建直观的Web界面
- **持久化存储**：向量数据库本地持久化，无需重复处理相同文档

## 技术栈

- **LangChain**: LLM应用开发框架
- **LangGraph**: Agent运行时
- **FAISS**: 向量数据库
- **DeepSeek**: 大语言模型
- **DashScope**: 阿里云模型服务（Embedding）
- **Streamlit**: Web框架
- **uv**: Python包管理工具

## 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/Chihong-Ng/RAG-Demo.git
   cd RAG-Demo
   ```

2. **初始化项目（使用uv）**
   ```bash
   uv init
   uv venv
   ```

3. **安装依赖**
   ```bash
   uv add streamlit langchain langchain-community langchain-core langchain-deepseek faiss-cpu dashscope python-dotenv pypdf
   ```

4. **设置环境变量**

   创建 `.env` 文件并添加你的API密钥：
   ```
   DEEPSEEK_API_KEY=您的DeepSeek_API_Key
   DASHSCOPE_API_KEY=您的DashScope_API_Key
   ```

5. **运行应用**
   ```bash
   uv run streamlit run main.py
   ```

6. **打开浏览器**

   访问 `http://localhost:8501` 开始使用应用

## 使用说明

1. 在侧边栏上传一个或多个PDF文件
2. 点击"提交并处理"按钮处理文档
3. 在主页面输入您的问题
4. AI将基于PDF内容回答问题

## 项目结构

```
RAG-Demo/
├── main.py           # 主程序入口
├── pyproject.toml    # uv项目配置
├── uv.lock          # 依赖锁定文件
├── .env             # 环境变量（需自行创建）
└── README.md        # 项目说明文档
```

## 注意事项

- 确保API密钥配置正确
- 首次使用需要上传PDF文件并处理
- 处理大文件可能需要一些时间
- 可以随时清除数据库重新开始
