# Import required packages

import collections
import copy
import json
import os
import re

import hazm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from cleantext import clean
from tqdm.notebook import tqdm
from transformers import BertConfig, BertModel, BertTokenizer


class Dataset(torch.utils.data.Dataset):
    """Create a PyTorch dataset"""

    def __init__(self, tokenizer, comments, targets=None, label_list=None, max_len=128):
        self.comments = comments
        self.targets = targets
        self.has_target = isinstance(targets, list) or isinstance(targets, np.ndarray)

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.label_map = (
            {label: i for i, label in enumerate(label_list)}
            if isinstance(label_list, list)
            else {}
        )

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])

        if self.has_target:
            target = self.label_map.get(
                str(self.targets[item]), str(self.targets[item])
            )

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        inputs = {
            "comment": comment,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
        }

        if self.has_target:
            inputs["targets"] = torch.tensor(target, dtype=torch.long)

        return inputs
