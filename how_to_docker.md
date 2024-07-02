# docker-jupyterの使い方
このリポジトリを`fork`してご使用ください。
雛形として自分の好きなライブラリやバージョンでコンテナを実行してください。
**自分の新しいリポジトリで管理する場合**
```shell
mkdir (任意のディレクトリ名)
cd (先ほどのディレクトリ名)
git clone git@github.com:ShotaArima/docker-jupyter.git
git remote remove origin
```
自分のgithubのサイトから新しくリポジトリを作成する。
ターミナルに戻り以下のコマンドを打ちます。
```shell
git remote add origin (新しく決めたリポジトリのsshまたはhttpのコード)
git push -u origin main
```

## ライブラリやバージョンの設定
このコンテナは、ベースイメージがminicondaを採用しています。
このディレクトリにある`enviornment.yml`ファイルへ移動し、編集してください。

## コンテナの起動
```shell
docker compose build # build
docker compose up -d # run
```
# 初めて環境構築する場合の手順

## (i). パスワードの設定
1. Google Colabを開く

2. 任意のパスワードを決める

3. 以下のコードでハッシュ化を行う
    ```python
    from notebook.auth import passwd
    passwd()
    ```
    <img width="420" alt="パスワードハッシュ化入力" src="https://github.com/ShotaArima/docker-jupyter/assets/130956497/36046f6c-d36a-4c25-a5b1-bd2d5b37d7ce">

    このコードを実行後、プロンプト下部にパスワード入力欄があるので入力する
    <img width="420" alt="パスワードハッシュ化確認" src="https://github.com/ShotaArima/docker-jupyter/assets/130956497/4ddb823c-5302-453d-9682-6a4d7f6a8ff4">

    ※ハッシュ化された文字列について、`$`があった場合、エスケープ処理を行う必要があります。`$`から`$$`に置換することでこの問題は解決することができました。

    

## (ii). ローカルでの実行方法（カーネルの設定）
1. Dockerを起動する
    ```shell
    docker compose up -d
    ```

2. VScodeでカーネルの選択を「別のカーネルを選択」する

3. `http://127.0.0.1:8888/`をURLに入力する

4. パスワードを入力する

5. pythonカーネルを選択する
6. `src/tmp.ipynb`pythonのコードが実行できるかどうかを確認する
    ```python
    print("Hello World!")
    ```
## (iii). matplotlibの日本語フォントの設定
1.  先ほどの`tmp.ipynb`を開いたまま、次のブロックに以下のコードを実行してください。
    ```python
    !find / -name "ttf"
    ```
    結果の中から、`opt/conda/envs/.../ttf`というディレクトリのものを探し、コピーします。
    `...`の部分は、自分の環境によって異なります。
2. dockerfileの一部を書き換えます。
    先ほどコピーしたディレクトリに`/ipaexg.ttf`をつけて内の以下の部分を書き換えます。
    {}の部分に先ほどのディレクトリを書いてください。
    ```dockerfile
    # RUN cp ipaexg00401/ipaexg.ttf {/opt/conda/envs/~/ttf}/ipaexg.ttf
    ```
3. `tmp.ipynb`に戻り以下のコードを実行してください。
    ```python
    !find / -name "matplotlibrc"
    ```
    結果の中から、`/opt/conda/.../matplotlibrc`というディレクトリを探し、コピーします。
    `...`の部分は、自分の環境によって異なります。
4. dockerfileの一部を書き換えます。
    先ほどのコピーしたディレクトリを{}の部分にそのまま置き換えます。
    ```dockerfile
    # RUN echo "font.family : IPAexGothic" >>  {/opt/conda/~/matplotlibrc}
    ```
5. dockerfile内の`#日本語フォントの設定`の下のコード全てのコメントアウトを解除します。
6. コンテナを起動し、問題がないことを確認してください。