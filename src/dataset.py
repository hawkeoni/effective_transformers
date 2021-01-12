from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.generate_listops import OPERATORS


class Vocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        tokens = ["[PAD", "[CLS]"] + list(map(str, range(10))) + OPERATORS + ["(", ")", "[", "]"]
        for i, token in enumerate(tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token

    def numericalize(self, tokens: List[str]) -> List[int]:
        return [self.word2idx[token] for token in tokens]

    def process_sample(self, sample: str) -> List[int]:
        arr = ["[CLS]"]
        arr.extend(sample.split())
        return self.numericalize(arr)


class ListOpsDataset(Dataset):

    def __init__(self, filename: str):
        self.vocab = Vocab()
        self.df = pd.read_csv(filename)
        self.df.Source = self.df.Source.apply(self.vocab.process_sample)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        x = self.df.iloc[index].Source
        y = [self.df.iloc[index].Target]
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        return x, y

    def __len__(self):
        return len(self.df)
