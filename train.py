#!/usr/bin/env python
# coding: utf-8

# Load dependencies
from utils import *
from dataset import *
from network import *

import numpy as np
import pandas as pd


import h5py
import os
import gc
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torchvision import transforms
import runai.ga.torch


# set random seed
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


train_path = "data/train.h5"
train_key = "train_dataset"
val_key = "val_dataset"
db_key = "db_dataset"

parser = argparse.ArgumentParser(
    description="Google Landmark Recognition Challenge - model training"
)

parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--optimizer", default="SGD")
parser.add_argument("--n_classes_subspace", default=12, type=int)
parser.add_argument("--n_samples_subspace", default=8, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_subspace", default=20, type=int)
parser.add_argument("--n_iter_subspace", default=200, type=int)
parser.add_argument("--img_size_tr", default=256, type=int)
parser.add_argument("--backbone", default="resnet101")
parser.add_argument("--subspace_backbone", default="resnet50")
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--crop_size", default=224, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--checkpoint_name", default="model_epoch4.pt")
parser.add_argument("--model_dir", default="models")
parser.add_argument("--tb_prefix", default="model_viz")
parser.add_argument("--log_freq", default=15, type=int)
parser.add_argument("--embedding_size", default=512, type=int)
parser.add_argument("--subspace_filename", default="h5/subspace_embeddings.hdf5")
parser.add_argument("--embeddings_key", default=None)
parser.add_argument("--ga_steps", default=15, type=int)

args = parser.parse_args()

logger = create_logger("train.log")

use_cuda = torch.cuda.is_available()
if use_cuda:
    logger.info("CUDA is available. Ready to start training.")


# implementation references:
# https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
# https://omoindrot.github.io/triplet-loss#offline-and-online-triplet-mining
def _pairwise_distances(embeddings, squared=False, use_l2=False):
    if use_l2:
        embeddings = F.normalize(embeddings, dim=1, p=2)
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)

    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances[distances < 0] = 0.0
    if not squared:
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16
        distances = (1.0 - mask) * torch.sqrt(distances)

    return distances


def _get_anchor_positive_mask(labels):
    """Return a 2D mask where mask[a, p] is True if
    a and p are distinct and have same label."""

    indices_equal = torch.eye(labels.size(0)).bool().cuda()
    indices_not_equal = ~indices_equal

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    return labels_equal & indices_not_equal


def _get_anchor_negative_mask(labels):
    """Return a 2D mask where mask[a, n] is True if
    a and n have distinct labels."""

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1)).cuda()

# triplet loss with online hard mining
def batch_hard_triplet_loss(embeddings, labels, squared=False, use_l2=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared, use_l2=use_l2)
    mask_anchor_positive = _get_anchor_positive_mask(labels).float()
    mask_anchor_negative = _get_anchor_negative_mask(labels).float()

    # get hardest positive
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # get hardest negative
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = (
        pairwise_dist * mask_anchor_negative
        + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    )
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    tl = torch.log1p(torch.exp(hardest_positive_dist - hardest_negative_dist))

    return tl.mean()


class TripletLoss(nn.Module):
    def __init__(self, squared=False, use_l2=False):
        super(TripletLoss, self).__init__()
        self.squared = squared
        self.use_l2 = use_l2

    def forward(self, x, labels):
        return batch_hard_triplet_loss(
            x, labels, squared=self.squared, use_l2=self.use_l2
        )


# training section
def init_optimizer(parameters, optimizer, lr):
    if optimizer == "Adam":
        return Adam(parameters, lr=lr)
    return SGD(parameters, lr=lr, momentum=0.9)


def build_subspace(
    subspace_filename,
    net,
    dataset,
    num_subspace,
    n_iter,
    logger,
    batch_size,
    log_freq,
    embeddings_key=None,
    subspace_backbone=None,
    num_workers=4,
):
    if embeddings_key:
        with h5py.File(subspace_filename, "r") as f:
            embeddings = torch.from_numpy(f[embeddings_key][:])
        logger.info("Embeddings for clustering loaded from disk.")
    else:
        logger.info("Start extracting embeddings for clustering...")
        subspace_dl = init_dataloader(
            dataset, num_workers, "subspace", logger, batch_size=batch_size
        )
        embeddings = extract_vectors(net, subspace_dl, logger, batch_size, log_freq)
        with h5py.File(subspace_filename, "a") as f:
            f.create_dataset(f"emb_{subspace_backbone}", data=embeddings.numpy())
        logger.info("Embeddings saved to disk.")

    class_centers_t, center_ids = get_class_centers(embeddings, dataset.labels)
    logger.info(f"Finished extracting {center_ids.shape[0]} class centers.")

    del embeddings
    gc.collect()

    clusters_t, _ = build_kmeans_cluster(
        class_centers_t, logger, K=num_subspace, n_iter=n_iter
    )

    return center_ids, clusters_t


def init_dataloader(
    dataset,
    num_workers,
    mode,
    logger,
    batch_size=None,
    batch_sampler=None,
    pin_memory=True,
):
    if batch_size == None:
        dl = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        sample_size = (
            len(dataset) // batch_sampler.batch_size
        ) * batch_sampler.batch_size
        logger.info(
            f"DataLoader for {mode} created. {len(dl)} batches, {sample_size} samples."
        )

    else:
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        logger.info(
            f"DataLoader for {mode} created. {len(dl)} batches, {len(dataset)} samples."
        )

    return dl


def train(
    train_dl,
    model,
    loss_fn,
    optimizer,
    epoch_idx,
    writer,
    batch_count,
    epochs,
    log_freq,
    logger,
):
    losses = AverageMeter()

    model.train()

    for batch_idx, (input_t, labels_t) in enumerate(train_dl):
        input_g = input_t.cuda(non_blocking=True)
        labels_g = labels_t.cuda(non_blocking=True)

        optimizer.zero_grad()
        output_g = model(input_g)
        loss = loss_fn(output_g, labels_g)
        losses.update(loss.item(), input_t.size(0))

        loss.backward()
        optimizer.step()

        writer.add_scalar("Training loss", loss, batch_count)
        batch_count += 1

        if (batch_idx + 1) % log_freq == 0 or (batch_idx + 1) == len(train_dl):
            logger.info(
                f"[E{epoch_idx}/{epochs}] Finished training batch No.{batch_idx+1}/{len(train_dl)}. "
                f"Current loss {losses.val:.4f}. Average loss {losses.avg:.4f}."
            )

    #     visualize the last batch of input images for each epoch
    writer.add_images("landmark_images", denormalize(input_t), epoch_idx)
    logger.info("Batch images logged in Tensorboard.")

    return losses.val


def validate(
    val_dl, model, loss_fn, epoch_idx, writer, batch_count, epochs, log_freq, logger
):
    losses = AverageMeter()

    model.eval()

    val_vecs = torch.zeros(len(val_dl.dataset), model.out_dim)

    with torch.no_grad():
        for batch_idx, (input_t, labels_t) in enumerate(val_dl):
            input_g = input_t.cuda(non_blocking=True)
            labels_g = labels_t.cuda(non_blocking=True)

            output_g = model(input_g)
            loss = loss_fn(output_g, labels_g)
            losses.update(loss.item(), input_t.size(0))

            val_vecs[
                batch_idx * len(input_t) : (batch_idx + 1) * len(input_t)
            ] = output_g.cpu()

            writer.add_scalar("Validation loss", loss, batch_count)
            batch_count += 1

            if (batch_idx + 1) % log_freq == 0 or (batch_idx + 1) == len(val_dl):
                logger.info(
                    f"[E{epoch_idx}/{epochs}] Finished validating batch No.{batch_idx+1}/{len(val_dl)}. "
                    f"Current loss {losses.val:.4f}. Average loss {losses.avg:.4f}."
                )

        return losses.avg, val_vecs


def main():
    model = LandmarkRecognitionNet(
        args.backbone, embedding_size=args.embedding_size, pooling="gem", use_fc=True
    ).cuda()
    logger.info("Recognition network initiated.")
    if use_cuda:
        logger.info("Using CUDA")
    optimizer = init_optimizer(model.parameters(), args.optimizer, args.lr)
    ga_optimizer = runai.ga.torch.optim.Optimizer(optimizer, steps=args.ga_steps)
    # scheduler = lr_scheduler.StepLR(ga_optimizer, step_size=2, gamma=0.1)

    loss_fn = TripletLoss().cuda()

    transform_tr = transforms.Compose(
        [
            transforms.RandomCrop(args.crop_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    transform_val = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_dataset = LandmarkDataset(
        train_path,
        train_key,
        args.img_size_tr,
        transform=transform_tr,
    )

    center_ids, clusters_t = build_subspace(
        args.subspace_filename,
        LandmarkRecognitionNet(args.subspace_backbone, extract_conv=True).cuda(),
        LandmarkDataset(train_path, train_key, args.crop_size, transform=transform_val),
        args.num_subspace,
        args.n_iter_subspace,
        logger,
        args.batch_size,
        args.log_freq,
        embeddings_key=args.embeddings_key,
        subspace_backbone=args.subspace_backbone,
        num_workers=args.num_workers,
    )

    val_dataset = LandmarkDataset(
        train_path,
        val_key,
        img_size=args.crop_size,
        transform=transform_val,
    )

    val_dl = init_dataloader(
        val_dataset,
        args.num_workers,
        "validation",
        logger,
        batch_size=args.batch_size,
    )

    db_dataset = LandmarkDataset(
        train_path,
        db_key,
        args.crop_size,
        transform=transform_val,
    )

    db_dl = init_dataloader(
        db_dataset,
        args.num_workers,
        "database",
        logger,
        batch_size=args.batch_size,
    )

    start_epoch = 1
    train_batch_count = 0
    val_batch_count = 0

    if args.checkpoint_name:
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(os.path.join(args.model_dir, args.checkpoint_name))
        start_epoch = checkpoint["epoch"] + 1
        train_batch_count = checkpoint["train_batch_count"]
        val_batch_count = checkpoint["val_batch_count"]
        best_gap_score = checkpoint["gap_score"]

        model.load_state_dict(checkpoint["model_state_dict"])
        ga_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Checkpoint loaded for epoch {start_epoch}.")
    else:
        best_gap_score = 0.0

    train_writer = init_tensorboard_writers(args.tb_prefix, "train")
    val_writer = init_tensorboard_writers(args.tb_prefix, "val")

    for epoch_idx in range(start_epoch, args.epochs + 1):
        logger.info(f"Starting epoch {epoch_idx} of {args.epochs}...")

        for i in range(start_epoch):
            subspace_batchsampler = ClusterBatchSampler(
                train_dataset.labels,
                center_ids,
                clusters_t,
                args.n_classes_subspace,
                args.n_samples_subspace,
            )
        logger.info("Batch sampler created.")

        train_dl = init_dataloader(
            train_dataset,
            args.num_workers,
            "training",
            logger,
            batch_size=None,
            batch_sampler=subspace_batchsampler,
        )

        train_loss = train(
            train_dl,
            model,
            loss_fn,
            ga_optimizer,
            epoch_idx,
            train_writer,
            train_batch_count,
            args.epochs,
            args.log_freq,
            logger,
        )
        logger.info(
            f"Training for E{epoch_idx} finished. Losses logged in Tensorboard."
        )
        train_batch_count += len(train_dl)

        val_loss, val_vecs = validate(
            val_dl,
            model,
            loss_fn,
            epoch_idx,
            val_writer,
            val_batch_count,
            args.epochs,
            args.log_freq,
            logger,
        )
        logger.info(
            f"Validation for E{epoch_idx} finished. Losses logged in Tensorboard."
        )
        val_batch_count += len(val_dl)

        logger.info("Extracting embeddings for database images...")
        db_vecs = extract_vectors(model, db_dl, logger, args.batch_size, args.log_freq)
        db_labels = db_dataset.labels
        db_centers, db_center_ids = get_class_centers(db_vecs, db_labels)

        center_scores, center_indices = get_topk_scores(db_centers, val_vecs, k=1)
        center_indices = center_indices.numpy().reshape(-1)
        center_scores = center_scores.numpy().reshape(-1)
        center_preds = db_center_ids[center_indices]

        scores, indices = get_topk_scores(db_vecs, val_vecs, k=1)
        indices = indices.numpy().reshape(-1)
        scores = scores.numpy().reshape(-1)
        preds = db_labels[indices]

        val_ids = val_dataset.annotations["id"].values
        val_true = pd.Series(val_dataset.labels, index=val_ids, name="y_true")
        val_true = val_true.map(lambda x: np.array([x]))

        nearest_preds = pd.DataFrame({"y_pred": preds, "conf": scores}, index=val_ids)
        nearest_center_preds = pd.DataFrame(
            {"y_pred": center_preds, "conf": center_scores}, index=val_ids
        )

        gap_score, landmark_acc, _ = compute_metrics(nearest_preds, val_true)
        gap_score_center, landmark_acc_center, _ = compute_metrics(
            nearest_center_preds, val_true
        )

        logger.info(
            f"[E{epoch_idx}] GAP score: {gap_score:.3f}, landmark accuracy: {landmark_acc:.2f}, "
            f"center GAP score: {gap_score_center:.3f}, center landmark accuracy: {landmark_acc_center:.2f}"
        )

        is_best = gap_score > best_gap_score
        best_gap_score = max(gap_score, best_gap_score)

        state = {
            "epoch": epoch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": ga_optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gap_score": gap_score,
            "landmark_acc": landmark_acc,
            "gap_score_center": gap_score_center,
            "landmark_acc_center": landmark_acc_center,
            "train_batch_count": train_batch_count,
            "val_batch_count": val_batch_count,
        }

        save_checkpoint(state, args.model_dir, logger, is_best=is_best)

    train_writer.flush()
    val_writer.flush()
    train_writer.close()
    val_writer.close()

    logger.info("Training has finished!")


if __name__ == "__main__":
    main()
