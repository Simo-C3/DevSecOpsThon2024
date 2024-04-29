# ベースイメージを指定
#FROM nginx:1.25
FROM nginx

# Nginsの設定をカスタマイズしてる
RUN rm /etc/nginx/conf.d/default.conf
COPY ./nginx.conf /etc/nginx/conf.d/nginx.conf

COPY ./server.crt /etc/nginx/ssl/server.crt
COPY ./server.key /etc/nginx/ssl/server.key
