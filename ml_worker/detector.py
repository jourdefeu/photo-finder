import os
import cv2
import numpy as np
import traceback
from insightface.app import FaceAnalysis
from insightface.utils import face_align

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

    def is_detect(self, input_path, output_path=None):
        """
        Находит лица на фото, возвращает true/false.
        """
        img = cv2.imread(input_path)
        if img is None:
            print(f"⚠️ Не удалось прочитать {input_path}")
            return False

        faces = self.app.get(img)

        if output_path not None:
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
            print(f"💾 Фото с обнаруженными лицами сохранено: {output_path}")

        return True

    def align_detected(self, input_path):
        """
        Получение вектора загружаемого пользователем фото.
        """
        img = cv2.imread(input_path)
        if img is None:
            print(f"⚠️ Не удалось прочитать {input_path}")
            return []  # return False

        faces = self.app.get(img)
        print(f"📸 {os.path.basename(input_path)} → найдено {len(faces)} лиц")

        aligned_faces_info = []

        for i, face in enumerate(faces):

            aligned_faces_info.append({
                "photo_id": os.path.splitext(os.path.basename(input_path))[0],
                "bbox": face.bbox.tolist(),
                "pose": tuple(face.pose) if face.pose is not None else (0,0,0),
                "embedding": face.embedding
            })

        print(f"✅ Всего найденных лиц: {len(aligned_faces_info)} в {os.path.basename(input_path)}")
        return aligned_faces_info
