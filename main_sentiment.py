# Import required packages

import collections
import copy
import json
import os
import re
from typing import Union

import hazm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from cleantext import clean

# from tqdm.notebook import tqdm
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from DatasetUtil import Dataset
from preprocess import cleaning

MODEL_NAME_OR_PATH = "HooshvareLab/bert-fa-base-uncased"


class SentimentModel(nn.Module):
    def __init__(self, config: dict):
        super(SentimentModel, self).__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME_OR_PATH, return_dict=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def create_data_loader(x, tokenizer, max_len, label_list, batch_size=16):
    dataset = Dataset(
        comments=x,
        # targets=y,
        tokenizer=tokenizer,
        max_len=max_len,
        label_list=label_list,
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def predict(model, text, tokenizer, max_len=128, batch_size=32):
    predictions = []
    prediction_probs = []
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    encoding = tokenizer.encode_plus(
        text,
        max_length=128,
        truncation=True,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_token_type_ids=True,
        return_attention_mask=True,
        padding="max_length",
        return_tensors="pt",  # Return PyTorch tensors
    )
    comments = []
    comments.append(text)
    label_list = ["-1", "0", "1"]
    test_data_loader = create_data_loader(
        comments, tokenizer, 128, label_list, batch_size=32
    )
    for dl in tqdm(test_data_loader, position=0):
        input_ids = dl["input_ids"]
        attention_mask = dl["attention_mask"]
        token_type_ids = dl["token_type_ids"]
        # compute predicted outputs by passing inputs to the model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # convert output probabilities to predicted class
        _, preds = torch.max(outputs, dim=1)

        predictions.extend(preds)
        prediction_probs.extend(F.softmax(outputs, dim=1))

    predictions = torch.stack(predictions).cpu().detach().numpy()
    prediction_probs = torch.stack(prediction_probs).cpu().detach().numpy()

    return predictions, prediction_probs


def sentiment_product(text):
    MODEL_NAME_OR_PATH = "HooshvareLab/bert-fa-base-uncased"

    text = cleaning(text)

    # pt_model = torch.load("bert_sentiment_v1.bin", map_location="cpu")
    pt_model = torch.load("bert_sentiment_v1.pt", map_location="cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    p1, p2 = predict(pt_model, text, tokenizer, 128, 32)
    return p1[0]


if __name__ == "__main__":
    text = "محصول خیلی خوب و کاربردی هست ولی قیمتش بالاست"
    sentiment_product(text)
