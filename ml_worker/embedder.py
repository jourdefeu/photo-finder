import faiss
import numpy as np
import json
import os
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceEmbeddingDatabaseFAISS:
    def __init__(self, embedding_dim=512, threshold=0.6):
        self.threshold = threshold
        self.embedding_dim = embedding_dim

        self.index = faiss.IndexFlatIP(embedding_dim)  # создаём FAISS-индекс для промежуточных векторов
        self.meta = []  # список метаданных (словарей)

    def add_from_aligned_info(self, aligned_faces_info):
        """
        Добавляет эмбеддинги из align_from_detected.
        """
        for face_info in aligned_faces_info:
            emb = np.array(face_info["embedding"], dtype=np.float32)
            emb /= np.linalg.norm(emb)          # нормализуем перед FAISS cosine similarity
            self.index.add(emb[np.newaxis, :])  # добавляем в FAISS
            self.meta.append({
                "photo_id": face_info["photo_id"],
                "bbox": face_info["bbox"],
                "pose": face_info["pose"]
            })
        logger.info(f"⬈ Сохранено {len(aligned_faces_info)} лиц в векторном представлении")

    def cluster_embeddings(self):
        """
        Быстрая кластеризация через FAISS:
        - для каждого вектора ищем соседей по косинусному сходству
        - объединяем, если similarity >= threshold
        - не связываем лица с одного фото
        """
        n = self.index.ntotal  # количество векторов в FAISS
        cluster_labels = [-1] * n
        cluster_id = 0

        # Для каждого лица ищем ближайших соседей
        emb_mat = np.array([self.index.reconstruct(i) for i in range(n)], dtype=np.float32)
        k = min(100, n)  # ограничим число соседей для ускорения

        for i in range(n):
            if cluster_labels[i] != -1:
                continue

            cluster_labels[i] = cluster_id
            emb = emb_mat[i].reshape(1, -1)

            # ищем ближайших соседей по FAISS
            sims, idxs = self.index.search(emb, k)
            sims = sims[0]
            idxs = idxs[0]

            for sim, j in zip(sims, idxs):
                if j == -1 or j == i:
                    continue
                if sim < self.threshold:
                    continue
                if self.meta[i]["photo_id"] == self.meta[j]["photo_id"]:
                    continue
                cluster_labels[j] = cluster_id

            cluster_id += 1

        # сгруппируем по ID
        clusters = defaultdict(list)
        for idx, c_id in enumerate(cluster_labels):
            clusters[c_id].append(idx)

        # усредненные векторы
        averaged_vectors = []
        cluster_metadata = []

        for rank, (c_id, indices) in enumerate(clusters.items(), start=1):
            cluster_vecs = [emb_mat[i] for i in indices]
            avg_vec = np.mean(cluster_vecs, axis=0)
            avg_vec /= np.linalg.norm(avg_vec)
            averaged_vectors.append(avg_vec)

            cluster_metadata.append({
                "user_id": f"{rank:05d}",  # '00001', '00002', ...
                "photo_ids": [self.meta[i]["photo_id"] for i in indices],    # ["img_001", "img_002"]
                "count": len(indices)      # количество лиц (векторов), вошедших в данный кластер
            })

        return averaged_vectors, cluster_metadata

    def save_database(self, save_dir):
        """
        Сохраняет усреднённые лица в FAISS и метаданные в JSON.
        """
        os.makedirs(save_dir, exist_ok=True)

        # кластеризация и усреднение
        avg_vecs, meta = self.cluster_embeddings()

        # создаём FAISS-индекс под усреднённые вектора
        dim = avg_vecs[0].shape[0]
        index = faiss.IndexFlatIP(dim)
        index.add(np.array(avg_vecs, dtype=np.float32))

        # сохраняем FAISS индекс
        faiss.write_index(index, os.path.join(save_dir, "faiss_index.idx"))

        # сохраняем метаданные (photo_ids, cluster_id и т.п.)
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # выводим статистику
        logger.info(f"Сохранено {len(avg_vecs)} усреднённых лиц в векторную БД ({save_dir})")



