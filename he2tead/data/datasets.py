from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TCGADataset(Dataset):

    def __init__(
        self,
        df,
        root,
        max_tiles=10_000
    ):
        super().__init__()
        self.filenames = list(root.glob('*.svs/features.npy'))
        self.ids = [str(path).split('/')[-2][:12] for path in self.filenames]
        self.filenames = np.array(self.filenames)[pd.Series(self.ids).isin(df.index).values]
        self.ids = np.array(self.ids)[pd.Series(self.ids).isin(df.index).values]
        self.df = df
        self.y = self.df.loc[self.ids, ['TEAD_500']].fillna(0).values.astype(float)
        self.max_tiles = max_tiles

    def __getitem__(self, i):
        x = np.load(self.filenames[i])
        x = x[:self.max_tiles, 3:]
        if x.shape[0] < self.max_tiles:
            x = np.concatenate([x, np.zeros((10_000 - x.shape[0], x.shape[1]))])
        y = self.y[i]
        return torch.Tensor(x), y

    def __len__(self):
        return len(self.ids)


class CPTACDataset(Dataset):

    def __init__(
        self,
        df,
        root,
        max_tiles=10_000
    ):
        super().__init__()
        self.filenames = list(root.glob('*.svs/features.npy'))
        self.slide_ids = [str(path).split('/')[-2].strip('.svs') for path in self.filenames]
        self.filenames = np.array(self.filenames)[pd.Series(self.slide_ids).isin(df.index).values]
        self.slide_ids = np.array(self.slide_ids)[pd.Series(self.slide_ids).isin(df.index).values]
        self.ids = np.array(['-'.join(s.split('-')[:2]) for s in self.slide_ids])
        self.df = df
        self.y = self.df.loc[self.slide_ids, ['TEAD_500']].fillna(0).values.astype(float)

    def __getitem__(self, i):
        x = np.load(self.filenames[i])
        x = x[:10_000, 3:]
        if x.shape[0] < self.max_tiles:
            x = np.concatenate([x, np.zeros((10_000 - x.shape[0], x.shape[1]))])
        y = self.y[i]
        return torch.Tensor(x), y

    def __len__(self):
        return len(self.ids)
