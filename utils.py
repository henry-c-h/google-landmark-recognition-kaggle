import logging
import gc
import os
import datetime
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter
import torch.linalg as LA


def create_logger(file_name, level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(file_name, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


class AverageMeter(object):
    """Computed and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_class_centers(embeddings, labels):
    """Returns the class centers of input embeddings by averaging and their class labels"""

    center_ids = np.unique(labels)

    label2idx = {landmark_id: idx for idx, landmark_id in enumerate(center_ids)}
    idx_labels = np.vectorize(label2idx.get)(labels)
    idx_labels = torch.from_numpy(
        idx_labels
    ).long()  # scatter_add_ expect int64 for indices arg

    indices = idx_labels.view(idx_labels.size(0), 1).expand(-1, embeddings.size(1))
    unique_idx, label_counts = indices.unique(dim=0, return_counts=True)

    # use scatter_add to sum values from embeddings of the same labels
    total = torch.zeros_like(unique_idx, dtype=torch.float32).scatter_add_(
        0, indices, embeddings
    )
    class_centers = total / label_counts.unsqueeze(1)

    del indices
    del total
    gc.collect()

    return class_centers, center_ids


def gem_pooling(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)  # add to .parameters() of module
        self.eps = eps

    def forward(self, x):
        return gem_pooling(x, p=self.p, eps=self.eps)


def l2_norm(x, eps=1e-6):
    return x / (LA.norm(x, ord=2, dim=1, keepdim=True) + eps).expand(x.size())


class L2Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super(L2Norm, self).__init__()
        self.eps = eps

    def forward(self, x):
        return l2_norm(x, eps=self.eps)


def init_tensorboard_writers(tb_prefix, mode):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    log_dir = os.path.join("runs", tb_prefix, time_str)

    writer = SummaryWriter(log_dir=log_dir + f"_{mode}")
    return writer


def save_checkpoint(state, out_dir, logger, is_best=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file_path = os.path.join(out_dir, f'model_epoch{state["epoch"]}.pt')
    logger.info(f'[E{state["epoch"]}] Saving checkpoint...')
    torch.save(state, file_path)

    if is_best:
        best_path = os.path.join(out_dir, "model_best.pt")
        shutil.copyfile(file_path, best_path)


def denormalize(img_batch):
    img_batch = img_batch.permute(0, 2, 3, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_batch = (img_batch * std + mean).permute(0, 3, 1, 2)
    return img_batch


def get_topk_scores(db_vecs, query_vecs, k=1):
    db_vecs = db_vecs.cuda()
    query_vecs = query_vecs.cuda()

    db_l2 = F.normalize(db_vecs, dim=1, p=2)
    query_l2 = F.normalize(query_vecs, dim=1, p=2)

    cosine_g = torch.matmul(query_l2, db_l2.t())
    top_k_scores, top_k_indices = torch.topk(cosine_g, k=k, dim=1)

    top_k_scores = top_k_scores.cpu()
    top_k_indices = top_k_indices.cpu()

    return top_k_scores, top_k_indices


def compute_metrics(y_pred, y_true, non_landmark_id=203094):
    # sort by score in descending order
    indices = np.argsort(-y_pred["conf"].values)
    y_pred_sorted = y_pred.take(indices)
    y_true_sorted = y_true[indices]

    df = y_pred_sorted.merge(y_true_sorted, left_index=True, right_index=True)
    df["result"] = df.apply(lambda row: row["y_pred"] in row["y_true"], axis=1).astype(
        int
    )
    df["precision"] = df["result"].cumsum() / (np.arange(len(df)) + 1)
    df["is_landmark"] = (~df["y_true"].isin([np.array([non_landmark_id])])).astype(int)
    n_landmarks = df["is_landmark"].sum()

    gap_score = (df["precision"] * df["result"]).sum() / n_landmarks
    landmark_df = df[df["is_landmark"] == 1]
    landmark_acc = landmark_df["result"].sum() / landmark_df.shape[0]

    df = df.reset_index()

    return gap_score, landmark_acc, df
