import pandas as pd
from pathlib import Path


def load_data():
    df = pd.read_parquet(
        "/lcrc/project/FastBayes/rayandrew/sched-cl/raw_data/alibaba-big-chunk/chunk-0.parquet",
        engine="fastparquet",
    )
    return df


df = load_data()
# take first N rows
print("BEFORE REDUCE", len(df))

# N_ROWS = 1_500_000
N_ROWS = 100_000
df = df.iloc[:N_ROWS]
# df["cpu_avg_pred"] = pd.cut(df["cpu_avg"], bins=4, labels=False)
# df["cpu_avg_pred"] = pd.cut(df["cpu_avg"], bins=4, labels=False)
# print(df["cpu_avg_pred"].unique())
print("AFTER REDUCE", len(df))
# print(df.instance_name)
# print(df["instance_name.1"])
print(df.head())
print(df.columns)
dest = Path(
    "/lcrc/project/FastBayes/rayandrew/sched-cl/raw_data/alibaba/chunk-0.parquet"
)
if dest.exists():
    dest.unlink()
df.to_parquet(
    dest,
    index=False,
)
