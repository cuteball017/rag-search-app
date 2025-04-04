import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone

# .env 読み込み
load_dotenv()

# API キー取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# チェック
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    st.error("❗ .env ファイルに API キーが設定されていません。")
    st.stop()

# ベクトルストア初期化
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# LLM + RAG チェーン
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ✅ 세션 초기화 함수
def reset_inputs():
    for key in ["lost_item", "brand", "lost_date", "lost_place", "lost_color", "features"]:
        if key in st.session_state:
             del st.session_state[key]
    st.rerun() 
    
# 페이지 설정
st.set_page_config(page_title="RAG 落とし物検索", page_icon="🔍")
st.title("🔍 落とし物 RAG 検索インターフェース")

# 입력 필드 (각 입력에 고유 key를 줌)
st.text_input("📦 紛失物の名前", key="lost_item")
st.text_input("🏷️ ブランド名", key="brand")
st.text_input("📅 紛失日", key="lost_date")
st.text_input("📍 紛失場所", key="lost_place")
st.text_input("🎨 色", key="lost_color")
st.text_area("🧷 特徴（詳細）", key="features")

# 🔄 リセット 버튼 (위에)
if st.button("🔄 リセット"):
    reset_inputs()

# 🔎 検索する 버튼 (아래에)
if st.button("🔎 検索する"):
    query_parts = [
        f"紛失物: {st.session_state.lost_item}" if st.session_state.lost_item else "",
        f"ブランド: {st.session_state.brand}" if st.session_state.brand else "",
        f"日付: {st.session_state.lost_date}" if st.session_state.lost_date else "",
        f"場所: {st.session_state.lost_place}" if st.session_state.lost_place else "",
        f"色: {st.session_state.lost_color}" if st.session_state.lost_color else "",
        f"特徴: {st.session_state.features}" if st.session_state.features else ""
    ]
    query = " / ".join(filter(None, query_parts))

    with st.spinner("📡 RAG 検索中..."):
        result = qa_chain.invoke({"query": query})
        st.success("✅ 類似するアイテムが見つかりました！")

        st.subheader("🔎 検索結果")
        st.write(result["result"])

        with st.expander("📄 参照された元文書を表示"):
            for i, doc in enumerate(result.get("source_documents", [])):
                st.markdown(f"**文書 {i+1}:**")
                st.code(doc.page_content, language="text")





# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import RetrievalQA
# from pinecone import Pinecone

# # .env 読み込み
# load_dotenv()

# # API キー取得
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# # チェック
# if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
#     st.error("❗ .env ファイルに API キーが設定されていません。")
#     st.stop()

# # ベクトルストア初期化
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(PINECONE_INDEX_NAME)
# vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# # LLM + RAG チェーン
# llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True
# )

# # Streamlit UI
# st.set_page_config(page_title="RAG 落とし物検索", page_icon="🔍")
# st.title("🔍 落とし物 RAG 検索インターフェース")

# # 入力フィールド
# lost_item = st.text_input("📦 紛失物の名前", key="lost_item")
# brand = st.text_input("🏷️ ブランド名", key="brand")
# lost_date = st.text_input("📅 紛失日", key="lost_date")
# lost_place = st.text_input("📍 紛失場所", key="lost_place")
# lost_color = st.text_input("🎨 色", key="lost_color")
# features = st.text_area("🧷 特徴（詳細）", key="features")

# # 👇 리셋 함수 정의
# def reset_fields():
#     for key in ["lost_item", "brand", "lost_date", "lost_place", "lost_color", "features"]:
#         if key in st.session_state:
#             del st.session_state[key]
#     st.experimental_rerun()

# # 🔍検索 ＆ 🔄リセット ボタン
# btn_col1, btn_col2 = st.columns([1, 4])

# with btn_col1:
#     search_clicked = st.button("🔎 検索する")

# with btn_col2:
#     st.button("🔄 リセット", on_click=reset_fields)

# # 検索処理
# if search_clicked:
#     query_parts = [
#         f"紛失物: {lost_item}" if lost_item else "",
#         f"ブランド: {brand}" if brand else "",
#         f"日付: {lost_date}" if lost_date else "",
#         f"場所: {lost_place}" if lost_place else "",
#         f"色: {lost_color}" if lost_color else "",
#         f"特徴: {features}" if features else ""
#     ]
#     query = " / ".join(filter(None, query_parts))

#     with st.spinner("📡 RAG 検索中..."):
#         result = qa_chain.invoke({"query": query})
#         st.success("✅ 類似するアイテムが見つかりました！")

#         # 結果表示
#         st.subheader("🔎 検索結果")
#         st.write(result["result"])

#         # ソース表示
#         with st.expander("📄 参照された元文書を表示"):
#             for i, doc in enumerate(result.get("source_documents", [])):
#                 st.markdown(f"**文書 {i+1}:**")
#                 st.code(doc.page_content, language="text")



# streamlit run "c:/Users/kwskm/OneDrive/바탕 화면/RAG/test-match.py"

