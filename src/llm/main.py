import pandas as pd
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import pprint
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import sqlite3
from streaming_callback_handler import StreamingCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

def load_model(model_path):
  # モデルの設定
  llm = LlamaCpp(
      model_path=model_path,
      n_gpu_layers=25,  # gpuに処理させるlayerの数
      stop=["Question:", "Answer:"],  # 停止文字列
      streaming=True,  # ストリーミング処理を行うか
      callbacks=[StreamingCallbackHandler()]
  )
  return llm

def chat(question, chain):
  response = chain.invoke(question)
  return response

def llm(prompt: str):
  # インデックスのパス
  index_path = "./storage"

  # モデルのパス
  model_path = "../model/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"


  # インデックスの読み込み
  index = FAISS.load_local(
      folder_path=index_path,
      embeddings=embedding_model,
      allow_dangerous_deserialization=True,
  )

  # プロンプトテンプレートの定義
  question_prompt_template = """あなたは親切で優しいアシスタントです。丁寧に、日本語でお答えください！
もし以下の情報が探している情報に関連していない場合は、そのトピックに関する自身の知識を用いて質問
に答えてください。

  {context}

  Question: {question}
  Answer: """

  # プロンプトの設定
  QUESTION_PROMPT = PromptTemplate(
      template=question_prompt_template,  # プロンプトテンプレートをセット
      input_variables=["context", "question"]  # プロンプトに挿入する変数
  )

  system_template="あなたは、質問者からの質問を{language}で回答するAIです。"
  human_template="質問者：{question}"
  system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


  # モデルの設定
  llm = load_model(model_path)

  retriever = index.as_retriever(search_kwargs={'k': 2})
  # memory = VectorStoreRetrieverMemory(retriever=retriever)
  # chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)
  # （RAG用）質問回答chainの設定
  chain = RetrievalQA.from_chain_type(
      llm=llm,
      retriever=index.as_retriever(
          search_kwargs={'k': 2}  # indexから上位いくつの検索結果を取得するか
      ),
      chain_type_kwargs={"prompt": QUESTION_PROMPT},  # プロンプトをセット
      chain_type="stuff",  # 検索した文章の処理方法
      return_source_documents=True  # indexの検索結果を確認する場合はTrue
  )
  response = chat(prompt,chain)
  # while True:
  #   question = input("入力してください：")

  #   # LLMの回答生成
  #   response = chat(question,chain)

  #   # # indexの検索結果を確認
  #   # for i, source in enumerate(response["source_documents"], 1):
  #   #     print(f"\nindex: {i}----------------------------------------------------")
  #   #     print(f"{source.page_content}")
  #   #     print("---------------------------------------------------------------")

  #   print(response)
  #   # # 回答を確認
  #   response_result = response["result"]
  #   print(f"\nAnswer: {response_result}")
