# Import required packages

import numpy as np
import pandas as pd
import hazm
from cleantext import clean
from tqdm import tqdm
import os
import re
import json
import copy
import collections

from transformers import BertConfig, BertTokenizer
from transformers import BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from langdetect import detect
from finglish import f2p
from transformers import pipeline
from preprocess import cleaning
from DatasetUtil import Dataset

# import Dataset
MODEL_NAME_OR_PATH = "HooshvareLab/bert-fa-base-uncased"


class SentimentModel(nn.Module):
    def __init__(self, config):
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
    print("tokens  are:")
    print(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)
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
    print(len(test_data_loader))
    for dl in tqdm(test_data_loader, position=0):
        print(dl)
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
    try:
        lang = detect(text)
    except:
        lang = "en"

    print(lang)
    if lang == "fa":
        text = cleaning(text)
        pt_model = torch.load("./bert_sentiment_v1.pt", map_location="cpu")
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        p1, p2 = predict(pt_model, text, tokenizer, 128, 32)
        print(p1[0])
        print(p2)
        return p1[0]
    elif lang == "en":

        classifier = pipeline(
            "text-classification",
            model="j-hartmann/sentiment-roberta-large-english-3-classes",
            return_all_scores=True,
        )
        p = classifier(text)
        print(p)
        predictions = []
        predictions = [item.get("score") for item in p[0]]
        print(predictions)
        label = predictions.index(max(predictions))
        return label

    else:
        text = f2p(text)
        print(text)
        text = cleaning(text)
        pt_model = torch.load("./bert_sentiment_v1.pt", map_location="cpu")
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        p1, p2 = predict(pt_model, text, tokenizer, 128, 32)
        print(p1[0])
        print(p2)
        return p1[0]


if __name__ == "__main__":
    text = "محصول خیلی خوب و کاربردی هست ولی قیمتش بالاست"
    text1 = "in laptop daghoone cheghad :("
    text2 = "in kar aalie"
    text3 = "cheghad ghashange"
    text4 = "wo so cute"
    text5 = "this is not bad"
    text6 = "exciting"
    text7 = "😀"
    text8 = ":("
    text9 = "its so forbidding!"
    text10 = "hi"
    text11 = "🤬"
    print(sentiment_product(text1))
