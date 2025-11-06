import os
import cv2
import numpy as np
import traceback
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ограничиваем CPU до ~70%
cpu_count = max(1, int(os.cpu_count() * 0.7))
os.environ["OMP_NUM_THREADS"] = str(cpu_count)

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
            
            # проверяем landmarks
            landmark = face.landmark_2d_5 if face.landmark_2d_5 is not None else None

            if landmark is not None:
                try:
                    # нормальное выравнивание через norm_crop
                    aligned_face = face_align.norm_crop(img, landmark=landmark)
                except Exception as e:
                    print(f"⚠️ Не удалось выровнять лицо {i+1}, fallback на bbox: {e}")
                    aligned_face = img[y1:y2, x1:x2]
            else:
                # fallback: просто crop по bbox
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

