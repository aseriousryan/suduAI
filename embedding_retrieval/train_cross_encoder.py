import pandas as pd

import logging
import argparse

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from typing import List

def load_dataset(file_path):
    df = pd.read_excel(file_path)
    dataset = []
    for idx, row in df.iterrows():
        pos_instance = InputExample(texts=[row['Table_Description'], 'Positive'], label=1)
        neg_instance = InputExample(texts=[row['Table_Description'], 'Negative'], label=0)

        dataset.append(pos_instance)
        dataset.append(neg_instance)

    return dataset

def train(
    train_dataset: List,
    val_dataset: List,
    model_save_path: str,
    batch_size: int = 32,
    epoch: int = 5
):
    logging.basicConfig(
        format='- %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[LoggingHandler()]
    )

    model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_dataset, name='cross_encoder_val')

    warmup_steps = int(len(train_dataloader) * 5 * 0.1)
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epoch,
        evaluation_steps=int(len(train_dataloader) / epoch),
        warmup_steps=warmup_steps,
        save_best_model=True,
        output_path=model_save_path
    )

if __name__ == '__main__':
    train_dataset = load_dataset('../data/Latest Training Data.xlsx')
    val_dataset = load_dataset('../data/Latest Test Data2.xlsx')
    train(
        train_dataset,
        val_dataset,
        './models/cross_encoder_01/'
    )