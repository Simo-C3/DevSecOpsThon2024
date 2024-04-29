import pandas as pd
import requests
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

with open("./config.yml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# 認証オブジェクトを作成
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# ログインフォームの表示
if st.session_state["authentication_status"] is None:
    authenticator.login()

# ログイン成功時の処理
if st.session_state["authentication_status"]:
    authenticator.logout()
    # st.switch_page("pages/garbage_list.py") リダイレクトではない
# ログイン失敗時のエラーメッセージ
elif st.session_state["authentication_status"] is False:
    st.error("ユーザー名・パスワードが間違っています。")
elif st.session_state["authentication_status"] is None:
    st.warning("ユーザー名とパスワードを入力してください。")

# st.write("# ゴミの一覧")
# ゴミの一覧API
# API_ENDPOINT = "https://jsonplaceholder.typicode.com/todos"
# バックエンドAPIからデータを取得
# response = requests.get(API_ENDPOINT)

# レスポンスが成功した場合
# if response.status_code == 200:
#     data = response.json()  # JSON形式のレスポンスをデコード
#     df = pd.DataFrame(data)  # 辞書を含むリストに変換してDataFrameに変換
#     aa = df["id"].unique()
#     st.write(df)  # DataFrameを表示
#     s = st.selectbox("ゴミの出し方を選択", aa, key="df")
#     ss = df[df["id"] == s]
#     st.table(ss)
# else:
#     st.error("Failed to fetch data from backend API")
