from detect import FaceDetector
import os

# инициализация
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
