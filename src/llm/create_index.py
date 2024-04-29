from langchain.document_loaders import CSVLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import pprint
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


index_path = "./storage"

# モデルの読み込み
# 実行するモデルの指定とキャッシュフォルダの指定
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)
# cache_folder = "./sentence_transformers"

loader = DirectoryLoader(path="./", loader_cls=CSVLoader, glob='*.csv')
# loader = CSVLoader(
#     './combined_tables.csv', encoding="utf-8")
docs = loader.load()

pprint.pprint(docs[:5])


# インデックスの作成
index = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model,
)
# インデックスの保存
index.save_local(
    folder_path=index_path
)
