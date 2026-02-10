import pandas as pd
import numpy as np
from pathlib import Path

def load_data(data_dir='data/split', dataset='ETTh1'):
    """データを読み込む（全モデル共通）"""
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / f'{dataset}_train.csv')
    val_df = pd.read_csv(data_dir / f'{dataset}_val.csv')
    test_df = pd.read_csv(data_dir / f'{dataset}_test.csv')
    return train_df, val_df, test_df