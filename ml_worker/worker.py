import os
from detector import FaceDetector        # класс-детект
from embedder import FaceEmbeddingDatabaseFAISS  # класс, где реализуется сравнение и усреднение

if __name__ == "__main__":
    detector = FaceDetector(device="cpu")
    db = FaceEmbeddingDatabaseFAISS(threshold=0.6)  # создаём базу для эмбеддингов

    input_dir = "data/photos/raw_uploads"
    detected_dir = "data/photos/detected_preview"
    vector_dir = "data/vectors"

    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)

    # проход по всем изображениям
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(detected_dir, filename)

            # -- детекция и рамки
            success = detector.is_detect(in_path, out_path)
            if success:
                print(f"💾 Фото с обнаруженными лицами сохранено: {out_path}")

                # -- выравнивание отдельных лиц и получение эмбеддингов
                aligned_faces_info = detector.align_detected(in_path)

                # -- добавляем эмбеддинги в векторную базу
                db.add_from_aligned_info(aligned_faces_info)

    # -- сохраняем обновлённую базу эмбеддингов
    db.save_database(vector_dir)
    print(f"✅ Векторная база успешно сохранена: {vector_dir}")