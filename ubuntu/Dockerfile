# GPUを使うためのドライバがインストールされているubuntu22.04をBaseレイヤにする
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
# コンテナを起動する際に必要最低限のパッケージをインストールする。
RUN apt update && apt install -y \
        wget \
        bzip2 \
        git \
        unzip \
        file \
        xz-utils \
        sudo \
        vim \
        python3 \
        python3-pip \
        locales \
        && locale-gen ja_JP.UTF-8

# 環境変数を設定。日本語ロケールを適用する。
ENV LANG=ja_JP.UTF-8

# ホストのボリュームをマウントするためのディレクトリ作成。docker-compose.ymlでvolumeアタッチしたい場合に使うとよい。
RUN mkdir /llm

WORKDIR llm

# requirements.txt をコピーして必要なライブラリをインストール。pipでインストールしたいものはrequirements.txtに書くとよい。
COPY ./ubuntu/requirements.txt /tmp/
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --no-cache-dir -r /tmp/requirements.txt

# RUN pip install --no-cache-dir -U pip setuptools wheel \
#     && CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install -U langchain-community
RUN pip install -U "huggingface_hub[cli]"
RUN pip install -U langchain-community
