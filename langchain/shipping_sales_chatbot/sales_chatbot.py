import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    # 定义 prompt
    prompt_template = """你是一名经验丰富的航运客服专员，请使用航运业的通用知识回答顾客的问题。若遇到不清楚的问题请如实告诉客户，并致以歉意。
    上下文为：{context}
    问题为： {question}
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    global SALES_BOT    
    SALES_BOT = RetrievalQA(retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                      search_kwargs={"score_threshold": 0.8}),
                            combine_documents_chain=qa_chain)

    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    print("ans===========", ans)
    if ans["source_documents"] or ans['result']:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "请稍等，我与业务部门确认一下这个问题。"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="航运客服咨询",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="localhost")

if __name__ == "__main__":
    # 初始化船运客服销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
