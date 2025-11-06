туцюзнimport os
import cv2
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
            return False

        faces = self.app.get(img)
        valid_faces = 0

        for i, face in enumerate(faces):
            if face.pose is None or face.landmark_2d_5 is None:
                continue

            yaw, pitch, roll = face.pose
            if abs(yaw) > self.yaw_threshold:
                continue

            aligned_face = face_align.warp_and_crop_face(
                img, face.landmark_2d_106,
                reference_3d=face_align.get_reference_facial_points(default_square=True)
            )
            out_file = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(input_path))[0]}_{i+1}.jpg"
            )
            cv2.imwrite(out_file, aligned_face)
            valid_faces += 1
            print(f"💾 Сохранено лицо: {out_file}")

        print(f"✅ Всего выровненных лиц: {valid_faces} в {os.path.basename(input_path)}")
        return valid_faces > 0

