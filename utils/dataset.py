import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.fcgr import FCGR
from typing import Literal

class FCGRDataset(Dataset):
    def __init__(self, data: pd.DataFrame, k: int, m_rna_col: str, mi_rna_col: str, label_col: str, dataset_type: Literal["Training", "Testing", "Validation"]):
        self.data = data
        self.k = k
        self.m_rna_col = m_rna_col
        self.mi_rna_col = mi_rna_col
        self.label_col = label_col
        self.dataset_type = dataset_type
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        m_rna = row[self.m_rna_col].replace("U", "T").upper()
        mi_rna = row[self.mi_rna_col].replace("U", "T").upper()
        label = row[self.label_col]

        # Generate FCGR matrix
        m_rna_fcgr = FCGR(sequence=m_rna, k=self.k).generate_fcgr()
        mi_rna_fcgr = FCGR(sequence=mi_rna, k=self.k).generate_fcgr()

        match self.k:
            case 3:
                assert m_rna_fcgr.shape == mi_rna_fcgr.shape == (8, 8)
            case 4:
                assert m_rna_fcgr.shape == mi_rna_fcgr.shape == (16, 16)
            case 5:
                assert m_rna_fcgr.shape == mi_rna_fcgr.shape == (32, 32)
            case 6:
                assert m_rna_fcgr.shape == mi_rna_fcgr.shape == (64, 64)
            case 7:
                assert m_rna_fcgr.shape == mi_rna_fcgr.shape == (128, 128)
            case 8:
                assert m_rna_fcgr.shape == mi_rna_fcgr.shape == (256, 256)
            case 9:
                assert m_rna_fcgr.shape == mi_rna_fcgr.shape == (512, 512)
            case _:
                raise ValueError(f"Invalid k_mer: {self.k_mer}")
        
        m_rna_tensor = torch.FloatTensor(m_rna_fcgr).unsqueeze(0)
        mi_rna_tensor = torch.FloatTensor(mi_rna_fcgr).unsqueeze(0)
        label_tensor = torch.LongTensor([label])[0]

        return m_rna_tensor, mi_rna_tensor, label_tensor