import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.nn import functional as F

from skimage import io
from skimage.transform import resize


class LandmarkDataset(Dataset):
    """Generic dataset that returns the image tensor, label and image id of a sample"""

    def __init__(self, annotations_path, file_key, img_size, transform=None):
        self.annotations = pd.read_hdf(annotations_path, file_key, mode="r")
        if "landmark_id" in self.annotations.columns:
            self.labels = self.annotations["landmark_id"].values
        self.img_size = (img_size, img_size)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        image = io.imread(sample["file_path"])
        image = resize(image, self.img_size, anti_aliasing=True)
        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        image_t = image.to(torch.float32)

        if isinstance(sample["landmark_id"], str):
            label_t = torch.tensor(int(sample["landmark_id"].split()[0]))
        else:
            label_t = torch.tensor(int(sample["landmark_id"]), dtype=torch.long)

        return (image_t, label_t)


# implementation borrowed from https://github.com/getkeops/keops/blob/master/pykeops/tutorials/kmeans/plot_kmeans_torch.py
def build_kmeans_cluster(embeddings, logger, K=10, n_iter=10):
    """Perform K Means clustering and return the cluster ids and the centroid of input embeddings"""

    embeddings = F.normalize(embeddings, dim=1, p=2)  # l2 norm
    embeddings_g = embeddings.cuda()

    # initialization of centroids
    centroids_init = embeddings_g[:K, :].detach().clone()

    x_i = embeddings_g.unsqueeze(1)
    for i in range(n_iter):
        c_j = centroids_init.unsqueeze(0)
        d_ij = ((x_i - c_j) ** 2).sum(-1)  # N * K matrix

        # cluster assignment
        clusters = d_ij.argmin(dim=1)

        # move centroids
        centroids = torch.zeros_like(centroids_init)
        centroids.scatter_add_(
            0, clusters[:, None].repeat(1, embeddings_g.shape[1]), embeddings_g
        )
        n_clusters = torch.bincount(clusters, minlength=K).type_as(centroids).view(K, 1)
        centroids /= n_clusters

    _, counts = clusters.unique(dim=0, return_counts=True)

    logger.info(
        f"Successfully formed {K} clusters. Largest cluster has {counts.max()} labels. Smallest cluster has {counts.min()} labels."
    )

    clusters_t = clusters.cpu()
    centroids_t = centroids.cpu()

    del embeddings_g
    torch.cuda.empty_cache()

    return clusters_t, centroids_t


# implementation borrowed from https://github.com/adambielski/siamese-triplet/blob/0c719f9e8f59fa386e8c59d10b2ddde9fac46276/datasets.py
class ClusterBatchSampler(BatchSampler):
    """A Pytorch Sampler subclass that randomly draws a fixed-sized batch from samples in the same cluster"""

    def __init__(self, labels, center_ids, clusters, n_classes, n_samples):
        self.labels = labels
        self.center_ids = center_ids
        self.clusters = clusters.numpy()
        self.n_clusters = len(np.unique(self.clusters))
        self.cluster_to_labels = {
            cluster: self.center_ids[np.nonzero(self.clusters == cluster)[0]]
            for cluster in range(self.n_clusters)
        }
        self.label_to_indices = {
            label: np.random.permutation(np.nonzero(self.labels == label)[0])
            for label in self.center_ids
        }
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_classes * self.n_samples
        self.n_dataset = len(self.labels)
        self.used_label_indices_count = {label: 0 for label in self.center_ids}

    def __iter__(self):
        while self.count + self.batch_size < self.n_dataset:
            cluster = np.random.choice(self.n_clusters, 1, replace=True)[0]
            classes = np.random.choice(
                self.cluster_to_labels[cluster], self.n_classes, replace=False
            )
            indices = []

            for cl in classes:
                indices.extend(
                    self.label_to_indices[cl][
                        self.used_label_indices_count[cl] : (
                            self.used_label_indices_count[cl] + self.n_samples
                        )
                    ]
                )

                self.used_label_indices_count[cl] += self.n_samples

                if self.used_label_indices_count[cl] + self.n_samples > len(
                    self.label_to_indices[cl]
                ):
                    np.random.shuffle(self.label_to_indices[cl])
                    self.used_label_indices_count[cl] = 0

            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size
