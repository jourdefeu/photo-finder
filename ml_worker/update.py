import os
import json
import numpy as np
import faiss
from embedder import FaceEmbeddingDatabaseFAISS  # твой класс для работы с FAISS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Параметры
SAVE_DIR = "data/vectors"       # FAISS + metadata
USERS_DIR = "data/photos/users" # куда сохраняем фото по пользователям
THRESHOLD = 0.6                  # косинусная дистанция для совпадения

def update_db(new_face_infos):
    """
    Добавляем новые фото в базу или обновляем существующих пользователей.

    new_face_infos: список словарей с ключами
        "embedding", "photo_id", "bbox", "pose", "path" (путь к фото)
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(USERS_DIR, exist_ok=True)

    # 1. Загружаем существующую базу
    db = FaceEmbeddingDatabaseFAISS(embedding_dim=512, threshold=THRESHOLD)
    faiss_path = os.path.join(SAVE_DIR, "faiss_index.idx")
    meta_path = os.path.join(SAVE_DIR, "metadata.json")
    db.meta = []  # инициализируем пустым списком
    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        db.index = faiss.read_index(faiss_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            db.meta = json.load(f)
        logger.info(f"Загружена существующая БД: {db.index.ntotal} кластеров из {SAVE_DIR}")
    else:
        # Создаём новый пустой индекс
        db.index = faiss.IndexFlatIP(512)
        logger.info(f"Создание новой БД в {SAVE_DIR}")

    # 2. Сначала сопоставляем каждое новое лицо с существующими пользователями
    unmatched_faces = []  # лица, которые не совпали с существующими пользователями
    
    for face_info in new_face_infos:
        emb = np.array(face_info["embedding"], dtype=np.float32)
        emb /= np.linalg.norm(emb)
        matched = False
        
        # ищем ближайшего существующего пользователя
        if db.index.ntotal > 0:
            query = np.expand_dims(emb, axis=0).astype(np.float32)
            sims, indices = db.index.search(query, k=1)
            sim = float(sims[0][0])
            best_idx = int(indices[0][0])
            
            if best_idx >= 0 and best_idx < len(db.meta) and sim >= THRESHOLD:
                # Найдено совпадение с существующим пользователем
                cluster_meta = db.meta[best_idx]
                old_embedding = np.array(db.index.reconstruct(best_idx), dtype=np.float32)
                n_old_faces = cluster_meta["count"]
                
                # Обновляем усредненный вектор (добавляем одно новое лицо)
                updated_embedding = (old_embedding * n_old_faces + emb) / (n_old_faces + 1)
                updated_embedding /= np.linalg.norm(updated_embedding)
                
                # Добавляем photo_id если его еще нет
                if face_info["photo_id"] not in cluster_meta["photo_ids"]:
                    cluster_meta["photo_ids"].append(face_info["photo_id"])
                
                # Обновляем count
                cluster_meta["count"] = n_old_faces + 1
                
                # Сохраняем обновленный вектор
                cluster_meta["_updated_embedding"] = updated_embedding.tolist()
                
                total_photos = len(cluster_meta["photo_ids"])
                logger.info(f"Обновлён пользователь: {cluster_meta['user_id']} (теперь {total_photos} фото)")
                matched = True
        
        if not matched:
            # Не совпало с существующими - добавим в список для кластеризации
            unmatched_faces.append(face_info)
    
    # 3. Кластеризуем только те лица, которые не совпали с существующими
    if unmatched_faces:
        temp_db = FaceEmbeddingDatabaseFAISS(embedding_dim=512, threshold=THRESHOLD)
        temp_db.add_from_aligned_info(unmatched_faces)
        new_averaged_vectors, new_clusters = temp_db.cluster_embeddings()
        
        # 4. Создаём новых пользователей из несовпавших кластеров
        for idx, cluster_meta in enumerate(new_clusters):
            new_vec = new_averaged_vectors[idx]
            new_user_id = max([int(m["user_id"]) for m in db.meta], default=0) + 1
            unique_photo_ids = list(set(cluster_meta["photo_ids"]))
            
            db.meta.append({
                "user_id": f"{new_user_id:05d}",
                "photo_ids": unique_photo_ids,
                "count": cluster_meta["count"],
                "_updated_embedding": new_vec.tolist()
            })
            logger.info(f"🆕 Добавлен новый пользователь: {new_user_id:05d} ({len(unique_photo_ids)} фото)")

    # 5. Пересоздаём FAISS с обновлёнными усреднёнными векторами
    embeddings = []
    for m in db.meta:
        if "_updated_embedding" in m:
            # Используем обновленный вектор
            embeddings.append(np.array(m["_updated_embedding"], dtype=np.float32))
            # Удаляем временное поле
            del m["_updated_embedding"]
        else:
            # Используем вектор из существующего индекса
            idx = db.meta.index(m)
            if idx < db.index.ntotal:
                embeddings.append(np.array(db.index.reconstruct(idx), dtype=np.float32))
            else:
                # Если индекс меньше метаданных (не должно быть, но на всякий случай)
                embeddings.append(np.zeros(512, dtype=np.float32))
    
    db.index = faiss.IndexFlatIP(512)
    if embeddings:
        db.index.add(np.array(embeddings, dtype=np.float32))

    # 6. Сохраняем FAISS и метаданные
    faiss.write_index(db.index, faiss_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(db.meta, f, ensure_ascii=False, indent=2)
    logger.info(f"FAISS и метаданные сохранены: {SAVE_DIR}")

    return db.meta

