# ベースイメージの指定
FROM continuumio/miniconda3:latest

# wgetとunzipをインストール
RUN apt-get update && apt-get install -y wget unzip

# environment.ymlをコピーして環境を作成
COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml

# デフォルトでAnaconda環境をアクティブ化
SHELL ["/bin/bash", "-c"]

# 作業ディレクトリを設定
WORKDIR /src/

# ポートの公開
EXPOSE 9002

# 日本語フォントの設定
# RUN wget -O font.zip "https://moji.or.jp/wp-content/ipafont/IPAexfont/ipaexg00401.zip"
# RUN unzip font.zip
# RUN cp ipaexg00401/ipaexg.ttf {/opt/conda/envs/~/ttf}/ipaexg.ttf
# RUN echo "font.family : IPAexGothic" >>  {/opt/conda/~/matplotlibrc}
