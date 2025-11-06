import os

cpu_limit = max(1, int(os.cpu_count() * 0.7))
os.environ["OMP_NUM_THREADS"] = str(cpu_limit)
os.environ["MKL_NUM_THREADS"] = str(cpu_limit)

import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align

class FaceDetector:
    def __init__(self, device="cpu", yaw_threshold=30):
        """
        RetinaFace (из insightface) для детекции лиц + выравнивание + pose filtering.
        Работает на CPU, если device="cpu".
        """
        ctx_id = 0 if device == "cuda" else -1
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id)
        self.yaw_threshold = yaw_threshold
        print(f"✅ FaceDetector initialized (device={device})")

    def detect_and_align(self, input_path, output_preview_dir=None, output_faces_dir=None, show=False):
        """
        Находит лица, фильтрует по повороту головы, выравнивает и сохраняет:
          - превью с рамками
          - выровненные лица
        """
        img = cv2.imread(input_path)
        if img is None:
            print(f"⚠️ Не удалось прочитать {input_path}")
            return False

        faces = self.app.get(img)
        print(f"📸 {os.path.basename(input_path)} → найдено {len(faces)} лиц")

        valid_faces = []
        for i, face in enumerate(faces):
            yaw, pitch, roll = face.pose
            if abs(yaw) > self.yaw_threshold:
                print(f"↩️ Пропущено лицо {i+1} — слишком повернуто (yaw={yaw:.1f}°)")
                continue

            if face.landmark_2d_5 is not None:
                aligned_face = face_align.norm_crop(img, landmark=face.landmark_2d_5)
                valid_faces.append({
                    "bbox": face.bbox.tolist(),
                    "pose": (yaw, pitch, roll),
                    "aligned": aligned_face
                })
            else:
                print(f"⚠️ Пропущено лицо {i+1} на {os.path.basename(input_path)} — ключевые точки не найдены")
                continue

            # выравнивание лица
            aligned_face = face_align.norm_crop(img, landmark=face.landmark_2d_5)
            valid_faces.append({
                "bbox": face.bbox.tolist(),
                "pose": (yaw, pitch, roll),
                "aligned": aligned_face
            })

            # сохраняем выровненное лицо
            if output_faces_dir:
                os.makedirs(output_faces_dir, exist_ok=True)
                face_filename = f"{os.path.splitext(os.path.basename(input_path))[0]}_face{i}.jpg"
                out_path = os.path.join(output_faces_dir, face_filename)
                cv2.imwrite(out_path, aligned_face)

            # рисуем рамки на превью
            if output_preview_dir:
                os.makedirs(output_preview_dir, exist_ok=True)
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"face {i+1} ({yaw:.1f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # сохраняем превью
        if output_preview_dir:
            preview_path = os.path.join(output_preview_dir, os.path.basename(input_path))
            cv2.imwrite(preview_path, img)

        if show:
            cv2.imshow("Detected & Aligned Faces", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return valid_faces
