import streamlit as st
from streamlit_chat import message
from llm import LLM_Chat
import os
import pandas as pd
import torch

if "chat" not in st.session_state:
    st.session_state.chat = []
if "is_llm_inited" not in st.session_state:
    st.session_state.is_llm_inited = False
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "llm" not in st.session_state:
    st.session_state.llm = False

st.title("AIに質問する")

model_files = [
    filename for filename in os.listdir("../model") if filename.split(".")[-1] in ["gguf"]
]
default_model_file_index = model_files.index("Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")
model_name = st.sidebar.selectbox(
    'LLMのモデル選択',
     model_files,
     index=default_model_file_index
)

@st.cache(allow_output_mutation=True)
def get_llm():
    llm = LLM_Chat()
    return llm

llm = get_llm()

if not st.session_state.is_llm_inited:
    st.session_state.selected_model = model_name
    st.session_state.is_llm_inited = llm.init_llm(st.session_state.selected_model)

if st.session_state.selected_model != model_name:
    # get_llm.cache()
    st.session_state.selected_model = model_name
    st.session_state.is_llm_inited = llm.init_llm(st.session_state.selected_model)

uploaded_file = st.sidebar.file_uploader("ファイル選択", type=["csv", "json", "txt"])
# アップロードされたファイルがある場合
if uploaded_file is not None:
    # ファイルを読み込んで表示するなどの処理を行う
    file_contents = uploaded_file.read()
    file_name = uploaded_file.name
    with open(f"./data/{file_name}", "wb") as f:
        f.write(file_contents)
    llm.add_doc(f"./data/{file_name}")

data_files = [
    filename for filename in os.listdir("./data") if filename.split(".")[-1] in ["csv", "json", "txt"]
]

st.sidebar.table(pd.DataFrame(data_files, columns=["ファイル名"]))

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    response = llm.chat(prompt)


    st.session_state.chat.append({"role": "assistant", "content": response["result"]})
    for i, source in enumerate(response["source_documents"], 1):
      st.markdown(source.page_content)
    #   print(f"\nindex: {i}----------------------------------------------------")
    #   print(f"{source.page_content}")
    #   print("---------------------------------------------------------------")


# if submitted:
#     st.session_state.past.append(user_message)
#     st.session_state.generated.append("AIが聞いてます")

# if st.session_state["generated"]:
#     for i in range(len(st.session_state["generated"])):
#         generated_message = st.session_state["generated"][i]
#         for char in generated_message:
#             st.write(char, end="", key=str(i))  # 1文字ずつ表示する
#             st.experimental_rerun()  # ストリーミング表示を実現するために再レンダリング
