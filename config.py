import torch
import multiprocessing as mp

CONFIG = {
    # Reproducibility
    "seed": 123,

    "use_test": False,

    # Dataset Details
    "raw_data_path": "./data/miraw.csv",
    "m_rna_col": "mRNA_Site_Transcript",
    "mi_rna_col": "mature_miRNA_Transcript",
    "label_col": "validation",

    #Training
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": mp.cpu_count() - 1,
}