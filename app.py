import cv2
import os
import streamlit as st
from ultralytics import YOLO

TMP_DIR_PATH = "tmp"

if not os.path.exists(TMP_DIR_PATH):
    os.makedirs(TMP_DIR_PATH)

st.title("Downy mildew detector")
pred_conf = st.sidebar.slider('検出感度', 0, 100, 50)

# ファイルのアップロード
image_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
weight_file = st.sidebar.file_uploader("Upload a weight (YOLOv5 .pt)", type=["pt"])

if image_file is not None:
    # アップロードした画像の保存
    file_path = os.path.join(TMP_DIR_PATH, image_file.name)
    with open(file_path, "wb") as f:
        f.write(image_file.getvalue())

    if weight_file is not None:
         # アップロードした画像の保存
        file_path2 = os.path.join(TMP_DIR_PATH, weight_file.name)
        with open(file_path2, "wb") as f2:
             f2.write(weight_file.getvalue())

        # YOLOの実行
        model = YOLO(file_path2)
        # model = YOLO  v5(file_path2)
        results = model.predict(file_path, save=True, conf = (100-pred_conf)/100)

        # 予測結果の描画
        img = cv2.imread(file_path)
        for point in results[0].boxes.xyxy:
            cv2.rectangle(img,
                          (int(point[0]), int(point[1])),
                          (int(point[2]), int(point[3])),
                          (0, 0, 255),
                          thickness=5)

    # 解析画像の保存
        analysis_img_path = os.path.join(TMP_DIR_PATH,
                                         f"analysis_{image_file.name}")
        cv2.imwrite(analysis_img_path, img)

    # 画像の表示
        st.image(analysis_img_path,
                 use_column_width=True)
