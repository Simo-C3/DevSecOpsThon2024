import glob

import pandas as pd
import requests
import streamlit as st

st.write("# つかわない")

uploaded_file = st.file_uploader("ファイル選択", type=["csv", "json", "txt"])

# アップロードされたファイルがある場合
if uploaded_file is not None:
    # ファイルを読み込んで表示するなどの処理を行う
    file_contents = uploaded_file.read()
    file_name = uploaded_file.name
    with open(f"./data/{file_name}", "wb") as f:
        f.write(file_contents)


# StreamlitアプリケーションのUI
user_input = st.text_input("入力:")
submit_button = st.button("送信")

# バックエンドAPIのエンドポイント
API_ENDPOINT = "http://example.com/api"

# 送信ボタンがクリックされたときの処理
if submit_button:
    # ユーザーからの入力を取得
    input_data = {"input": user_input}

    # APIにPOSTリクエストを送信
    response = requests.post(API_ENDPOINT, json=input_data)

    # レスポンスを処理
    if response.status_code == 200:
        st.success("APIへのリクエストが成功しました")
        result = response.json()
        st.write("APIからのレスポンス:", result)
    else:
        st.error("APIへのリクエストが失敗しました")
