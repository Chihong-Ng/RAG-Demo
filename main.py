import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=dashscope_api_key
)

def load_pdf(files):
    """加载PDF文件并返回Document对象列表"""
    docs = []
    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        pdf_loader = PyPDFLoader(tmp_path)
        docs.extend(pdf_loader.load())
        
        os.remove(tmp_path)
    
    return docs

def get_chunks(docs):
    """文档切片"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def vector_store(chunks):
    """转换成向量并保存到本地数据库"""
    vectors_store = FAISS.from_documents(chunks, embedding=embeddings)
    vectors_store.save_local("faiss_db")

def check_database_exists():
    """检查FAISS数据库是否存在"""
    return os.path.exists("faiss_db") and os.path.exists("faiss_db/index.faiss")

@tool
def retrieve_context(query: str):
    """检索相关信息来回答问题。"""
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

def get_conversational_chain(ques):
    """大模型问答"""
    llm = ChatDeepSeek(model="deepseek-chat", api_key=DeepSeek_API_KEY)
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是AI助手，请根据提供的上下文回答问题，确保提供所有细节，如果答案不在上下文中，请说"答案不在上下文中"，不要提供错误的答案"""
        ),
        ("human", "{context}\n\n问题: {question}")
    ])

    chain = (
        {"context": retrieve_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(ques)
    st.write(" 回答: ", response)

def main():
    st.set_page_config("LangChain RAG Demo")
    st.header("LangChain RAG Demo")

    col1, col2 = st.columns([3, 1])

    with col1:
        if not check_database_exists():
            st.warning("请先上传并处理PDF文件")

    with col2:
        if st.button("清除数据库"):
            try:
                import shutil
                if os.path.exists("faiss_db"):
                    shutil.rmtree("faiss_db")
                st.success("数据库已清除")
                st.rerun()
            except Exception as e:
                st.error(f"清除失败: {e}")

    user_question = st.text_input(
        "请输入问题",
        placeholder="例如：这个文档的主要内容是什么？",
        disabled=not check_database_exists()
    )
    
    if user_question:
        if check_database_exists():
            with st.spinner("AI正在分析文档..."):
                try:
                    get_conversational_chain(user_question)
                except Exception as e:
                    st.error(f"生成回答时出错: {str(e)}")
                    st.info("请重新处理PDF文件")
        else:
            st.error("请先上传并处理PDF文件！")

    with st.sidebar:
        st.title("文档管理")

        if check_database_exists():
            st.success("数据库状态：已就绪")
        else:
            st.info("状态：等待上传PDF")

        st.markdown("---")

        pdf_doc = st.file_uploader(
            "📎 上传PDF文件",
            accept_multiple_files=True,
            type=["pdf"],
            help="支持上传多个PDF文件"
        )

        if pdf_doc:
            st.info(f"已选择 {len(pdf_doc)} 个文件")
            for i, pdf in enumerate(pdf_doc, 1):
                st.write(f"{i}. {pdf.name}")

        process_button = st.button(
            "提交并处理",
            disabled=not pdf_doc,
            use_container_width=True
        )

        if process_button:
            if pdf_doc:
                with st.spinner("正在处理PDF文件..."):
                    try:
                        docs = load_pdf(pdf_doc)
                        if not docs:
                            st.error("无法从PDF中提取文本，请检查文件是否有效")
                            return

                        text_chunks = get_chunks(docs)
                        st.info(f"文本已分割为 {len(text_chunks)} 个片段")

                        vector_store(text_chunks)

                        st.success("PDF处理完成！现在可以开始提问了")
                        st.balloons()
                        st.rerun()

                    except Exception as e:
                        st.error(f"处理PDF时出错: {str(e)}")
            else:
                st.warning("请先选择PDF文件")

        with st.expander(" 使用说明"):
            st.markdown("""
            **步骤：**
            1. 上传一个或多个PDF文件
            2. 点击"提交并处理"处理文档
            3. 在主页面输入您的问题
            4. AI将基于PDF内容回答问题
            
            **提示：**
            - 支持多个PDF文件同时上传
            - 处理大文件可能需要一些时间
            - 可以随时清除数据库重新开始
            """)

if __name__ == '__main__':
    main()
