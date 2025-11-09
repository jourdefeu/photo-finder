# принимает фото, извлекает embedding, ищет совпадения
import os
import json
import faiss
import numpy as np
from detector import FaceDetector             # класс-детект

def vectorize_face(input_path):               # ./путь/img_334.jpg
    detector = FaceDetector(device="cpu")

    vector_dir = "data/vectors"               # путь к FAISS базе
    users_dir = "data/photos/users"           # путь к фоткам
    temporary_dir = "data/photos/temporary"   # путь к временному хранилищу фоток, отправляемых пользователями

    os.makedirs(temporary_dir, exist_ok=True)

    if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # -- детекция лица
        success = detector.is_detect(input_path)

        if not success:
            print("Лицо не найдено на изображении")
            return None

        # -- выравнивание лица и получение эмбеддинга
        aligned_face_info = detector.align_detected(input_path)

        emb = np.array(aligned_face_info["embedding"], dtype=np.float32)
        emb /= np.linalg.norm(emb)   # нормализуем для cosine similarity

        # -- удаление исходного фото
        os.remove(input_path)

        # -- загрузка FAISS индекса
        faiss_index_path = os.path.join(vector_dir, "faiss_index.idx")

        if not os.path.exists(faiss_index_path):
            print("❌ FAISS база не найдена")
            return None

        index = faiss.read_index(faiss_index_path)

        # -- загрузка метаданных
        metadata_path = os.path.join(vector_dir, "metadata.json")

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # -- поиск ближайшего вектора
        sims, idxs = index.search(emb.reshape(1, -1), k=1)
        best_sim = float(sims[0][0])
        best_idx = int(idxs[0][0])
        best_meta = meta[best_idx]

        print(f"✅ Найден ближайший кластер с similarity={best_sim:.4f}")
        print(f"→ Метаданные: {best_meta}")

        # -- проверка порога
        threshold = 0.6
        if best_sim < threshold:
            print("⚠️ Совпадений выше порога не найдено")
            return None

        # -- получаем user_id из метаданных
        user_id = best_meta["user_id"]

        # -- путь к папке с фото пользователя
        user_folder = os.path.join(users_dir, f"user_{user_id}")

        # -- список всех фото в папке
        user_photos = []
        if os.path.exists(user_folder):
            for fname in os.listdir(user_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                    user_photos.append(os.path.join(user_folder, fname))
        else:
            print(f"⚠️ Папка {user_folder} не найдена")

        # -- возвращаем результат
        return {
            "similarity": best_sim,
            "cluster_meta": best_meta,
            "user_id": best_meta["user_id"],
            "user_folder": user_folder,
            "user_photos": user_photos
        }