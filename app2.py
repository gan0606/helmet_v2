import streamlit as st
import shutil
import cv2
from PIL import Image
from ultralytics import YOLO
import os

# pyplotを使用する際に注記が出ないようにする文
st.set_option("deprecation.showPyplotGlobalUse", False)

# 関数化する
def main():
    # タイトル
    st.title("ヘルメット着用促進システムの開発")
    st.write("最終更新日: 2024/5/17")

    # サイドバーのmenu
    menu = ["イントロダクション", "検出結果・展望", "動画の物体検出", "物体検出マシン"]
    # サイドバーの作成
    chosen_menu = st.sidebar.selectbox(
        "menu選択", menu
    )

    # ファイルの設定

    # 検出対象の画像
    object_file = "train_imgs_with_bbox.jpg"
    # テストデータの結果
    result_file = "detecting2.jpg"
    # modelの読み込み
    model_file = "runs/detect/train/weights/best.pt"


    # 読み込めているかを確認
    is_object_file = os.path.isfile(object_file)
    is_result_file = os.path.isfile(result_file)
    is_model_file = os.path.isfile(model_file)

    # printで出力すると、ターミナルに出る
    # st.writeだとブラウザ上に出る
    print(is_object_file)
    print(is_result_file)
    print(is_model_file)
    
    

    # menuの中身
    # 分析の概要
    if chosen_menu == "イントロダクション":
        st.subheader("開発の概要")
        st.write("YOLOv8モデルをファインチューニングして画像の中から物体を検出できるAIを開発しました。")
        st.write("**YOLOv8について**")
        st.write(
            """
            YOLOv8は、リアルタイム物体検出に特化した深層学習モデルです。速度と精度の面で最先端の性能を実現しています。
            主な応用先としては、自動運転・監視カメラ・ドローンなどがあります。
        """
        )
        st.write("**ファインチューニングについて**")
        st.write(
            """
            ファインチューニングとは、公開されている学習済みのモデル(AI)を特定のタスクやデータセットに微調整する機械学習の手法です。
            ゼロからモデルを学習するよりも、短時間でモデルを特定のタスクに適応させることができます。

        """
        )
        st.subheader("開発の目的")
        st.write("・AIを活用したヘルメット着用自動判別システムで、作業員の安全確保と生産性向上を実現すること")
        st.write("・ヘルメット着用自動判別システムで、現場監督の巡視時間と労力を大幅に削減すること")

        st.subheader("データセットの内容")
        st.write(
            """
        モデルの訓練に使用したデータセットには、建設現場や製造現場などの職場で安全ヘルメットを着用している人物の画像と、
        着用していない人物の画像が合わせて7035枚(訓練情報: 5269枚, テスト情報: 1766枚)含まれています。
        また27039件の詳細なアノテーション情報も含まれています。
        """
        )
        st.write(
            """
        アノテーション情報とは画像に対して付加された補足情報のことです。
        データセットには、ヘルメットを着用した頭部(helmet)と、ヘルメットを着用していない頭部(head)、人物(person)の位置とサイズのアノテーション情報が含まれています。
        """
        )
        st.write(
            """
        5269枚の訓練情報を8:2の割合で訓練データと検証データに分割して使用しました。
            """
            )
        st.write("Roboflowより出典 Hard Hat Workers Dataset. https://public.roboflow.com/object-detection/hard-hat-workers")
        st.write(" ")
        st.write(" ")
        st.write("**訓練情報の一部**")
        st.write("訓練画像の一部にアノテーション情報を描画しました。")
        # 画像の表示
        image_object = Image.open(object_file)
        st.image(image_object)
       
    # 検出の結果
    elif chosen_menu == "検出結果・展望":
        st.subheader("検出結果")
        # 結果の表示
        image_result = Image.open(result_file)
        st.image(image_result)
        st.write(" ")
        st.write(" ")

        # 結果についての説明
        st.write(
            """
        開発したAIは、未知のデータにおいて、
        人物(person)の検出精度が2%と低かった一方、
        ヘルメット(97%)と頭部(93%)の検出精度に関しては高い性能を発揮することを示しました。
        """)
        st.write("上記の画像はテストデータのうち9枚を可視化したものです。")
        st.write(
            """
        人物(person)の検出精度が低かった原因としては、
        helmet, headに対してpersonのアノテーション情報が少なかったことが考えられます。
            """)
        st.write(" ")
        st.subheader("展望")
        st.write("仮に実務でこのようなAIの開発を行う際は、モデルの学習データを増やしたり、モデルを改良したりすることで、これらの状況における検出精度向上を目指します。")
        st.write("具体的には、下記のことを考えています。")
        st.write("・ヘルメットを被っていない作業員が後ろを向いている時や他の作業員と重なっているときの画像や映像を追加で収集する。")
        st.write("・helmet・head・personのアノテーションデータの数を均等化する。") 

    # 動画によるデモ
    elif chosen_menu == "動画の物体検出":
        st.subheader("動画の物体検出")
        st.write("開発したAIで動画からhelmet・head・personを検出します。")
        st.write(" ")
        st.write(" ")
        st.video("inference2_movies/sample4_inference2.mp4")
        st.write("pixabayより出典 https://pixabay.com/ja/videos")
        st.write(" ")
        st.write(
            """
            動画からヘルメットをかぶっている作業員とかぶっていない作業員を検出しましたが、未検出時間や誤検出がありました。
            特に、ヘルメットを被っていない作業員が後ろを向いている時や他の作業員と重なっているときに、未検出や誤検出が多く発生しました。
            仮に実務でこのようなAIの開発を行う際は、モデルの学習データを増やしたり、モデルを改良したりすることで、これらの状況における検出精度向上を目指します。
            """)
        st.write("具体的には、下記のことを考えています。")
        st.write("・ヘルメットを被っていない作業員が後ろを向いている時や他の作業員と重なっているときの画像や映像を追加で収集する。")
        st.write("・helmet・head・personのデータ数を均等化する。")


    elif chosen_menu == "物体検出マシン":
        st.subheader("物体検出マシン")
        st.write("訓練済みのAIでアップロードされた画像の中からhelmet・head・personを検出します。")
        # 空白行
        st.write("")
        # ラジオボタンの作成
        img_source = st.radio("画像のソースを選択してください", ("画像をアップロード", "カメラで撮影"))

        # 画像のアップロード
        if img_source == "画像をアップロード":
            # ファイルをアップロード
            img_file = st.file_uploader("画像を選択してください", type=["png", "jpg"])
        # カメラで撮影する場合
        elif img_source == "カメラで撮影":
            # カメラ撮影
            img_file = st.camera_input("カメラで撮影")

        # 推定の処理
        # img_fileが存在する場合に処理を進める
        if img_file is not None:
            # 特定の処理が行われていることを知らせる
            with st.spinner("検出中です..."):
                # 画像ファイルを開く
                img = Image.open(img_file)
                # 画面に画像を表示
                st.subheader("アップロードされた画像")
                st.image(img, caption="検出対象画像", width=480)

                # 空白行
                st.write("")

                # 予測
                model = YOLO(model_file).cpu()
                results = model(img)
                os.makedirs("result", exist_ok=True)
                for r in results:
                    img = r.plot()
                    cv2.imwrite("result/detect_0.jpg", img)

                # 結果の表示
                st.subheader("検出結果")
                result_path = "result/detect_0.jpg"
                result_img = cv2.imread(result_path)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                shutil.rmtree("result")
                st.image(result_img, caption="検出済み画像", width=480)


               

# streamlitを実行したときにmain()を実行するという表記
if __name__ == "__main__":
    main()
