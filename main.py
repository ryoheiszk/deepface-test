import sys
import os
from deepface import DeepFace
import cv2
import shutil
import time

TEMP_DIR = "tmp_face_img"

def extract_and_save_face(face, img_path, output_path):
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

def extract_face_from_target(img_path):
    start = time.time()
    faces = DeepFace.extract_faces(
        img_path=img_path,
        detector_backend="retinaface",
        align=True
    )

    if len(faces) != 1:
        raise ValueError(f"Target image must contain exactly one face. Found {len(faces)} faces.")

    target_path = os.path.join(TEMP_DIR, "target.jpg")
    extract_and_save_face(faces[0], img_path, target_path)
    print(f"Target detection time: {time.time() - start:.2f}s")

    return target_path

def calculate_face_similarities(target_img_path, group_img_path):
    target_face_path = extract_face_from_target(target_img_path)

    start = time.time()
    faces = DeepFace.extract_faces(
        img_path=group_img_path,
        detector_backend="retinaface",
        align=True
    )
    print(f"Group detection time: {time.time() - start:.2f}s")

    start = time.time()
    for i, face in enumerate(faces):
        temp_path = os.path.join(TEMP_DIR, f"temp_{i}.jpg")
        extract_and_save_face(face, group_img_path, temp_path)

        result = DeepFace.verify(
            img1_path=target_face_path,
            img2_path=temp_path,
            model_name="VGG-Face",
            enforce_detection=False,
        )

        distance = result["distance"]
        new_path = os.path.join(TEMP_DIR, f"{distance:.3f}_{i}.jpg")
        os.rename(temp_path, new_path)
    print(f"Verification time: {time.time() - start:.2f}s")

def main():
    if os.path.exists(TEMP_DIR):
        for file in os.listdir(TEMP_DIR):
            if file != '.gitkeep':
                file_path = os.path.join(TEMP_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    else:
        os.makedirs(TEMP_DIR)

    calculate_face_similarities(target_img_path=sys.argv[1], group_img_path=sys.argv[2])

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total time: {time.time() - start:.2f}s")
