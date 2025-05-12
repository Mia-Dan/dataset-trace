import pandas as pd
from pathlib import Path


def load_data():
    df = pd.read_parquet(
        "/lcrc/project/FastBayes/rayandrew/sched-cl/raw_data/alibaba/chunk-0.parquet",
        engine="fastparquet",
    )
    return df


df = load_data()
print(df.head())
print(df.columns)
