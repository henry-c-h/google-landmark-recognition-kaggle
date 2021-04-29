#!/usr/bin/env python
# coding: utf-8

from utils import *
from dataset import *
from network import *
import argparse

import numpy as np
import pandas as pd

import h5py
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms


# set random seed
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

test_file = "data/test.h5"
test_key = "test_private_df"
database_key = "db_df"
non_landmark_key = "non_l_df"

parser = argparse.ArgumentParser(
    description="Google Landmark Recognition Challenge - model testing"
)

parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--img_size", default=224, type=int)
parser.add_argument("--backbone", default="resnet101")
parser.add_argument("--embedding_size", default=512, type=int)
parser.add_argument("--model_dir", default="models")
parser.add_argument("--checkpoint_name", default="model_best.pt")
parser.add_argument("--emb_file", default="h5/embeddings.hdf5")
parser.add_argument("--load_emb", default=0, type=int)
parser.add_argument("--db_key", default="database")
parser.add_argument("--nl_key", default="non_landmark")
parser.add_argument("--log_freq", default=15, type=int)
parser.add_argument("--results_file", default="results_df")

args = parser.parse_args()

logger = create_logger("test.log")


def main():
    model = LandmarkRecognitionNet(
        args.backbone, embedding_size=args.embedding_size, pooling="gem", use_fc=True
    )
    model.cuda()
    model.eval()

    logger.info("Loading pretrained model...")
    checkpoint = torch.load(os.path.join(args.model_dir, args.checkpoint_name))
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded")

    transform_eval = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    db_ds = LandmarkDataset(
        test_file, database_key, args.img_size, transform=transform_eval
    )
    db_dl = DataLoader(
        db_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True
    )
    logger.info(
        f"Database dataloader: {len(db_dl.dataset)} samples, {len(db_dl)} batches"
    )

    test_ds = LandmarkDataset(
        test_file, test_key, args.img_size, transform=transform_eval
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(
        f"Test dataloader: {len(test_dl.dataset)} samples, {len(test_dl)} batches"
    )

    nl_ds = LandmarkDataset(
        test_file, non_landmark_key, args.img_size, transform=transform_eval
    )
    nl_dl = DataLoader(
        nl_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True
    )
    logger.info(
        f"Non-landmark dataloader: {len(nl_dl.dataset)} samples, {len(nl_dl)} batches"
    )

    if args.load_emb:
        with h5py.File(args.emb_file, "r") as f:
            db_emb = torch.from_numpy(f[args.db_key][:])
            nl_emb = torch.from_numpy(f[args.nl_key][:])
        logger.info("Database and non-landmark embeddings loaded from disk")
    else:
        logger.info("Start extracting features for database dataset...")
        db_emb = extract_vectors(model, db_dl, logger, args.batch_size, args.log_freq)
        logger.info("Start extracting features for non-landmark dataset...")
        nl_emb = extract_vectors(model, nl_dl, logger, args.batch_size, args.log_freq)
        with h5py.File(args.emb_file, "a") as f:
            f.create_dataset(args.db_key, data=db_emb.numpy())
            f.create_dataset(args.nl_key, data=nl_emb.numpy())
        logger.info("Database and non-landmark embeddings saved to disk")

    logger.info("Start extracting features for test dataset...")
    test_emb = extract_vectors(model, test_dl, logger, args.batch_size, args.log_freq)

    conf_scores = []
    nearest_preds = []
    nearest_indices = []
    for test_batch in test_emb.split(args.batch_size):
        scores, indices = get_topk_scores(db_emb, test_batch, k=1)
        scores = scores.view(-1)
        indices = indices.view(-1)

        preds = db_ds.labels[indices]
        nl_scores, _ = get_topk_scores(nl_emb, test_batch, k=5)
        nl_score = torch.mean(nl_scores, dim=1)
        final_score = (scores - nl_score).numpy()

        conf_scores.extend(final_score)
        nearest_preds.extend(preds)
        nearest_indices.extend(indices.view(-1).numpy())

    y_true = pd.Series(test_ds.labels, index=test_ds.annotations["id"], name="y_true")
    y_true = y_true.map(lambda x: np.array([int(i) for i in x.split()]))
    preds_df = pd.DataFrame(
        {"y_pred": nearest_preds, "conf": conf_scores, 'nearest': nearest_indices}, index=test_ds.annotations["id"]
    )

    gap_score, landmark_acc, df = compute_metrics(preds_df, y_true)

    df = df.astype({"y_true": "str"})
    df.to_hdf(test_file, key=args.results_file, mode="a")

    logger.info(
        f"Prediction finished. GAP score: {gap_score:.3f}, landmark accuracy: {landmark_acc:.2f}."
    )


if __name__ == "__main__":
    main()
