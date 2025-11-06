from detect import FaceDetector
import os

# инициализация
detector = FaceDetector(device="cpu")

input_dir = "try/photos/raw"
output_dir = "try/photos/detected"
os.makedirs(output_dir, exist_ok=True)

# проход по всем изображениям
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        success = detector.detect_and_draw(in_path, out_path)
        if success:
            print(f"💾 Сохранено: {out_path}")
