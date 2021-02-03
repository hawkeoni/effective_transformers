import re
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from effective_transformers.generate_listops import OPERATORS


class Vocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        tokens = ["[PAD", "[CLS]"] + list(map(str, range(10))) + OPERATORS + ["[", "]"]
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
        self.df = pd.read_csv(filename, sep="\t")

    def process_sample(self, sample: str) -> str:
        """
        See
        https://github.com/google-research/long-range-arena/issues/6
        """
        # remove brackets and replace spaces
        sample = sample.replace("(", "").replace(")", "")
        sample = re.sub(r"\s+", " ", sample)
        return sample

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        x: str = self.df.Source.iloc[index]
        x: str = self.process_sample(x)
        x_vec: List[int] = self.vocab.process_sample(x)
        y: List[int] = [self.df.iloc[index].Target]
        x_vec: torch.LongTensor = torch.LongTensor(x_vec)
        y_vec: torch.LongTensor = torch.LongTensor(y)
        return x_vec, y_vec

    def __len__(self):
        return len(self.df)
