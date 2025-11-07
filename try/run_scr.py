import os
from detect import FaceDetector        # твой класс-детект
from vec import FaceEmbeddingDatabase  # класс, где реализуешь сравнение и усреднение

if __name__ == "__main__":
    detector = FaceDetector(device="cpu")
    db = FaceEmbeddingDatabase(threshold=0.6)  # создаём базу для эмбеддингов

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

            # -- детекция и рамки
            success = detector.detect_and_draw(in_path, out_path)
            if success:
                print(f"💾 Фото с обнаруженными лицами сохранено: {out_path}")

                # -- выравнивание и сохранение отдельных лиц
                # ------ для каждого исходного фото будет своя подпапка с выровненными лицами
                # -- получение эмбеддингов
                aligned_dir_for_file = os.path.join(aligned_dir, os.path.splitext(filename)[0])
                os.makedirs(aligned_dir_for_file, exist_ok=True)

                aligned_faces_info = detector.align_from_detected(in_path, aligned_dir_for_file)

                # -- добавляем эмбеддинги в векторную базу
                db.add_from_aligned_info(aligned_faces_info)

    # -- сохраняем обновлённую базу эмбеддингов
    db.save_database("try/vector_db")
    print("✅ Векторная база успешно сохранена: try/vector_db")
