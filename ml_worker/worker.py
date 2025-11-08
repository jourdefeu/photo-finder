import os
import shutil
import json
from detector import FaceDetector        # класс-детект
from embedder import FaceEmbeddingDatabaseFAISS  # класс, где реализуется сравнение и усреднение

def save_user_photos(cluster_metadata, raw_photos_dir, users_dir):
    """
    raw_photos_dir — путь к исходным фото (input_dir)
    users_dir — куда складываем по пользователям (кластерам)
    cluster_metadata — результат cluster_embeddings
    """
    os.makedirs(users_dir, exist_ok=True)

    for user_idx, cluster in enumerate(cluster_metadata, 1):
        # можно использовать cluster_id или просто user_1, user_2...
        user_folder = os.path.join(users_dir, f"user_{user_idx}")
        os.makedirs(user_folder, exist_ok=True)

        # чтобы не копировать одно фото несколько раз
        seen_photos = set()

        for photo_id in cluster["photo_ids"]:
            if photo_id in seen_photos:
                continue
            seen_photos.add(photo_id)

            src_path = os.path.join(raw_photos_dir, f"{photo_id}.jpg")  # или .png, проверку можно добавить
            if os.path.exists(src_path):
                shutil.copy2(src_path, user_folder)
            else:
                print(f"⚠️ Фото {src_path} не найдено")

    print(f"✅ Скопированы фото по {len(cluster_metadata)} пользователям в {users_dir}")

if __name__ == "__main__":
    detector = FaceDetector(device="cpu")
    db = FaceEmbeddingDatabaseFAISS(threshold=0.6)  # создаём базу для эмбеддингов

    input_dir = "data/photos/raw_uploads"
    detected_dir = "data/photos/detected_preview"
    vector_dir = "data/vectors"
    users_dir = "data/photos/users"
    cluster_metadata_dir = "data/vectors/metadata.json"

    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)
    os.makedirs(users_dir, exist_ok=True)

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

    with open(cluster_metadata_dir, "r", encoding="utf-8") as f:
        cluster_metadata = json.load(f)

    save_user_photos(cluster_metadata, input_dir, users_dir)
