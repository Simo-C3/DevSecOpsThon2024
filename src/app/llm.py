import pandas as pd
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import torch
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gc

class LLM_Chat:
  def __init__(self):
    # gc.collect()
    # torch.cuda.empty_cache()
    self.index = None
    self.llm = None
    self.chain = None
    self.index_path = "./storage"

  def __del__(self):
    del self.index
    del self.llm
    del self.chain
    gc.collect()
    torch.cuda.empty_cache()
    print("LLM_Chat object is deleted")

  def add_doc(self, file_path):
    loader = TextLoader(file_path)
    doc=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
    doc = text_splitter.split_documents(doc)
    self.index.add_documents(doc)

  def load_model(self,model_name:str):
    # モデルの設定
    self.llm = LlamaCpp(
        model_path=f"../model/{model_name}",
        n_gpu_layers=35,  # gpuに処理させるlayerの数
        stop=["Question:", "Answer:"],  # 停止文字列
        streaming=True,  # ストリーミング処理を行うか
        n_ctx=2048,
        verbose=True,
        callbacks=[StreamingCallbackHandler()]
    )

  def chat(self, question):
    response = self.chain.invoke(question)

    print(response)

    return response

  def init_llm(self, model_name:str):
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    # インデックスの読み込み
    self.index = FAISS.load_local(
      folder_path=self.index_path,
      embeddings=embedding_model,
      allow_dangerous_deserialization=True,
    )
    # プロンプトテンプレートの定義
    question_prompt_template = """あなたはゴミの捨て方を答える親切で優しいアシスタントです。出力は丁寧な日本語で出してください。英語での回答はしないでください。
    また、出力文字数は100文字以内にしてください。
    contextの品名とquestionの対象が同じものを示している場合はcontextに従って回答してください。もし、contextとquestionの対象が同じものを示していない場合は、contextを無視してquestionに回答してください。
    収集の曜日が分かる場合は曜日も答えてください。収集の曜日は”:”の右側に記載されています。


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
    self.load_model(model_name)

    retriever = self.index.as_retriever(search_kwargs={'k': 2})
    # memory = VectorStoreRetrieverMemory(retriever=retriever)
    # chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)
    # （RAG用）質問回答chainの設定
    self.chain = RetrievalQA.from_chain_type(
        llm=self.llm,
        retriever=self.index.as_retriever(
          search_type="similarity_score_threshold",
          search_kwargs={'k': 2,"score_threshold": 0.7}  # indexから上位いくつの検索結果を取得するか
        ),
        chain_type_kwargs={"prompt": QUESTION_PROMPT},  # プロンプトをセット
        chain_type="stuff",  # 検索した文章の処理方法
        return_source_documents=True  # indexの検索結果を確認する場合はTrue
    )

    return True

  # def chat(prompt: str):

  #   response = chat(prompt,chain)
    # while True:
    #   question = input("入力してください：")

    #   # LLMの回答生成
    #   response = chat(question,chain)

    #   # # indexの検索結果を確認

    #   print(response)
    #   # # 回答を確認
    #   response_result = response["result"]
    #   print(f"\nAnswer: {response_result}")
