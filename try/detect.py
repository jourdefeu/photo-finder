import os
import cv2
import numpy as np
import traceback
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ----- ограничить CPU -----

# ----- попробовать пойти путем 1) обрезки, 2) выравнивания -----
# --------------- не рубить сразу landmark ---------------

class FaceDetector:
    def __init__(self, device="cpu", yaw_threshold=30):
        """
        RetinaFace (из insightface) для детекции лиц.
        Работает на CPU, если device="cpu".
        """
        ctx_id = 0 if device == "cuda" else -1
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id)
        self.yaw_threshold = yaw_threshold
        print(f"✅ FaceDetector initialized (device={device})")

    def detect_and_draw(self, input_path, output_path):
        """
        Находит лица на фото и сохраняет копию с нарисованными рамками.
        """
        img = cv2.imread(input_path)
        if img is None:
            print(f"⚠️ Не удалось прочитать {input_path}")
            return False

        faces = self.app.get(img)
        print(f"📸 {os.path.basename(input_path)} → найдено {len(faces)} лиц")

        # рисуем рамки вокруг лиц
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"face {i+1}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # сохраняем изображение с рамками
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return True

    def align_from_detected(self, input_path, output_path):
        """
        Фильтрует лица по повороту головы, выравнивает и сохраняет выровненные лица.
        """
        img = cv2.imread(input_path)
        if img is None:
            print(f"⚠️ Не удалось прочитать {input_path}")
            return []  # return False

        faces = self.app.get(img)
        aligned_faces_info = []

        for i, face in enumerate(faces):
            # bbox для лица
            x1, y1, x2, y2 = face.bbox.astype(int)
            
            try:
                # ✅ предпочтительный способ: использовать встроенное выравнивание InsightFace
                if hasattr(face, "aligned_face") and face.aligned_face is not None:
                    aligned_face = face.aligned_face
                # ⚙️ если нет aligned_face, пробуем выровнять вручную
                elif face.landmark_2d_5 is not None:
                    landmark = face.landmark_2d_5.astype("float32")
                    aligned_face = face_align.norm_crop(img, landmark)
                else:
                    # fallback: просто crop по bbox
                    aligned_face = img[y1:y2, x1:x2]
            except Exception as e:
                print(f"⚠️ Не удалось выровнять лицо {i+1}, fallback на bbox: {e}")
                aligned_face = img[y1:y2, x1:x2]

            out_file = os.path.join(
                output_path,
                f"{os.path.splitext(os.path.basename(input_path))[0]}_{i+1}.png"
            )
            cv2.imwrite(out_file, aligned_face)
            print(f"💾 Сохранено лицо: {out_file}")

            aligned_faces_info.append({
                "bbox": face.bbox.tolist(),
                "pose": tuple(face.pose) if face.pose is not None else (0,0,0),
                "aligned_path": out_file,
                "embedding": face.embedding
            })

        print(f"✅ Всего выровненных лиц: {len(aligned_faces_info)} в {os.path.basename(input_path)}")
        return aligned_faces_info

if __name__ == "__main__":
    detector = FaceDetector(device="cpu")

    input_dir = "try/photos/raw"
    detected_dir = "try/photos/detected"
    aligned_dir = "try/photos/aligned"

    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)

    # проход по всем изображениям
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(detected_dir, filename)

            # детекция и рамки
            success = detector.detect_and_draw(in_path, out_path)
            if success:
                print(f"💾 Фото с обнаруженными лицами сохранено: {out_path}")

                # выравнивание и сохранение отдельных лиц
                # для каждого исходного фото будет своя подпапка с выровненными лицами
                aligned_dir_for_file = os.path.join(aligned_dir, os.path.splitext(filename)[0])
                os.makedirs(aligned_dir_for_file, exist_ok=True)
                detector.align_from_detected(in_path, aligned_dir_for_file)
