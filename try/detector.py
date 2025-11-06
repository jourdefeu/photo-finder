import os
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align

class FaceDetector:
    def __init__(self, device="cpu"):
        """
        RetinaFace (из insightface) для детекции лиц.
        Работает на CPU, если device="cpu".
        """
        ctx_id = 0 if device == "cuda" else -1
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id)
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
