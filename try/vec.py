import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import uuid
import json
import os

class FaceEmbeddingDatabase:
    def __init__(self, threshold=0.6):
        """
        threshold — порог схожести: чем выше, тем строже.
        Рекомендуется 0.5–0.65 для ArcFace.
        """
        self.threshold = threshold
        self.embeddings = []   # список numpy-векторов
        self.meta = []         # метаданные (список словарей)

    def add_from_aligned_info(self, aligned_faces_info):
        """
        Добавляет эмбеддинги из align_from_detected
        photo_id — уникальный ID исходного фото
        """
        for face_info in aligned_faces_info:
            emb = np.array(face_info["embedding"], dtype=np.float32)
            self.embeddings.append(emb)
            self.meta.append({
                "photo_id": face_info["photo_id"],
                "bbox": face_info["bbox"],
                "pose": face_info["pose"],
                "aligned_path": face_info["aligned_path"]
            })
        print('⬈ Эмбеддинги сохранены.')
            
    def _pairwise_similarities(self):
        """Вычисляет попарные косинусные схожести."""
        if len(self.embeddings) < 2:
            return np.zeros((len(self.embeddings), len(self.embeddings)))
        emb_mat = np.stack(self.embeddings)
        return cosine_similarity(emb_mat)
    
    def cluster_embeddings(self):
        """
        Объединяет похожие вектора в группы (лица одного человека),
        при этом не связывает вектора с одного и того же фото.
        """
        n = len(self.embeddings)
        sim_matrix = self._pairwise_similarities()

        # хранение: индекс -> ID кластера
        cluster_labels = [-1] * n
        cluster_id = 0

        for i in range(n):
            if cluster_labels[i] != -1:
                continue

            # создаем новый кластер
            cluster_labels[i] = cluster_id
            for j in range(i + 1, n):
                if cluster_labels[j] != -1:
                    continue

                same_photo = (
                    self.meta[i]["photo_id"] == self.meta[j]["photo_id"]
                )
                if same_photo:
                    continue  # не связываем лица с одного фото

                similarity = sim_matrix[i, j]
                if similarity >= self.threshold:
                    cluster_labels[j] = cluster_id

            cluster_id += 1

        # сгруппируем по ID
        clusters = defaultdict(list)
        for idx, c_id in enumerate(cluster_labels):
            clusters[c_id].append(idx)

        # создаем усредненные векторы
        averaged_vectors = []
        cluster_metadata = []
        for c_id, indices in clusters.items():
            cluster_vecs = [self.embeddings[i] for i in indices]
            avg_vec = np.mean(cluster_vecs, axis=0)
            avg_vec /= np.linalg.norm(avg_vec)  # L2 нормализация
            averaged_vectors.append(avg_vec)

            cluster_metadata.append({
                "cluster_id": str(uuid.uuid4()),
                "photo_ids": [self.meta[i]["photo_id"] for i in indices],
                "aligned_paths": [self.meta[i]["aligned_path"] for i in indices],
            })

        return averaged_vectors, cluster_metadata

    def save_database(self, save_dir):
        """Сохраняет базу векторов и метаданных."""
        os.makedirs(save_dir, exist_ok=True)
        avg_vecs, meta = self.cluster_embeddings()

        np.save(os.path.join(save_dir, "face_vectors.npy"), np.array(avg_vecs))
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"✅ Сохранено {len(avg_vecs)} усреднённых лиц в {save_dir}")
