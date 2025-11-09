import os
import shutil
import json
from detect import FaceDetector                # твой класс-детект
from vec import FaceEmbeddingDatabaseFAISS     # класс, где реализуешь сравнение и усреднение

def save_user_photos(cluster_metadata, raw_photos_dir, users_dir):
    """
    raw_photos_dir — путь к исходным фото (input_dir)
    users_dir — куда складываем по пользователям (кластерам)
    cluster_metadata — результат cluster_embeddings
    """
    os.makedirs(users_dir, exist_ok=True)

    for _, cluster in enumerate(cluster_metadata, 1):
        user_id = cluster["user_id"]
        user_folder = os.path.join(users_dir, f"user_{user_id}")
        os.makedirs(user_folder, exist_ok=True)

        # чтобы не копировать одно фото несколько раз
        seen_photos = set()

        for photo_id in cluster["photo_ids"]:
            if photo_id in seen_photos:
                continue
            seen_photos.add(photo_id)

            # поддержка форматов jpg/jpeg/png/webp
            found = False
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                src_path = os.path.join(raw_photos_dir, f"{photo_id}{ext}")
                if os.path.exists(src_path):
                    shutil.copy2(src_path, user_folder)
                    found = True
                    break
            if not found:
                print(f"⚠️ Фото {photo_id} не найдено в {raw_photos_dir}")

    print(f"Фото {len(cluster_metadata)} пользователей сохранены в {users_dir}")

if __name__ == "__main__":
    detector = FaceDetector(device="cpu")
    db = FaceEmbeddingDatabaseFAISS(threshold=0.6)  # создаём базу для эмбеддингов

    input_dir = "try/photos/raw"
    detected_dir = "try/photos/detected"
    vector_dir = "try/vector_db"
    users_dir = "try/photos/users"
    cluster_metadata_dir = "try/vector_db/metadata.json"

    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)
    os.makedirs(users_dir, exist_ok=True)

    # проход по всем изображениям
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(detected_dir, filename)

            # -- детекция и рамки
            success = detector.detect_and_draw(in_path, out_path)
            if success:
                # -- выравнивание и сохранение отдельных лиц
                # -- получение эмбеддингов
                # aligned_dir_for_file = os.path.join(aligned_dir, os.path.splitext(filename)[0])
                # os.makedirs(aligned_dir_for_file, exist_ok=True)

                aligned_faces_info = detector.align_detected(in_path)

                # -- добавляем эмбеддинги в векторную базу
                db.add_from_aligned_info(aligned_faces_info)

    # -- сохраняем обновлённую базу эмбеддингов
    db.save_database(vector_dir)
    print(f"✅ Векторная база успешно сохранена: {vector_dir}")

    with open(cluster_metadata_dir, "r", encoding="utf-8") as f:
        cluster_metadata = json.load(f)

    save_user_photos(cluster_metadata, input_dir, users_dir)

