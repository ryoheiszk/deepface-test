import sys
import os
from deepface import DeepFace
import cv2
import time

TEMP_DIR = "tmp_face_img"

def get_target_face_paths(target_dir):
    """ターゲットディレクトリから顔画像のパスを取得"""
    face_paths = []
    for filename in os.listdir(target_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            face_paths.append(os.path.join(target_dir, filename))
    return face_paths

def extract_and_save_face(face, img_path, output_path):
    """グループ画像から検出した顔を切り出して保存"""
    face_area = face["facial_area"]
    img = cv2.imread(img_path)
    x, y = face_area["x"], face_area["y"]
    w, h = face_area["w"], face_area["h"]

    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(img.shape[1] - x, w + 2 * margin_x)
    h = min(img.shape[0] - y, h + 2 * margin_y)

    face_crop = img[y:y+h, x:x+w]
    cv2.imwrite(output_path, face_crop)

def extract_all_faces(group_img_path):
    """グループ画像から全ての顔を抽出して保存"""
    print("Extracting faces...")
    start = time.time()

    faces = DeepFace.extract_faces(
        img_path=group_img_path,
        detector_backend="yolov8",
        align=True
    )

    face_paths = []
    for i, face in enumerate(faces):
        face_path = os.path.join(TEMP_DIR, f"face_{i}.jpg")
        extract_and_save_face(face, group_img_path, face_path)
        face_paths.append(face_path)

    print(f"Extracted {len(faces)} faces in {time.time() - start:.2f}s")
    return face_paths

def process_faces(target_dir, extracted_face_paths):
    """抽出された顔画像それぞれについて類似度を計算"""
    target_face_paths = get_target_face_paths(target_dir)
    print(f"Found {len(target_face_paths)} target face images")

    start = time.time()
    for i, face_path in enumerate(extracted_face_paths):
        # 全てのターゲット画像と比較
        similarities = []
        for target_path in target_face_paths:
            result = DeepFace.verify(
                img1_path=target_path,
                img2_path=face_path,
                model_name="ArcFace",
                enforce_detection=False,
            )
            similarities.append(result["distance"])

        # 最も類似度が高い値を計算
        best_similarity = min(similarities)

        # 結果を含むファイル名で保存
        new_path = os.path.join(TEMP_DIR, f"{best_similarity:.3f}_{i}.jpg")
        os.rename(face_path, new_path)

        print(f"Face {i}: Best similarity = {best_similarity:.3f}")

    print(f"Verification time: {time.time() - start:.2f}s")

def main():
    # TEMPディレクトリの初期化
    if os.path.exists(TEMP_DIR):
        for file in os.listdir(TEMP_DIR):
            if file != '.gitkeep':
                file_path = os.path.join(TEMP_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    else:
        os.makedirs(TEMP_DIR)

    start_total = time.time()

    # 顔の抽出
    extracted_face_paths = extract_all_faces(sys.argv[2])

    # 抽出された顔の処理
    process_faces(sys.argv[1], extracted_face_paths)

    print(f"Total time: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    main()
