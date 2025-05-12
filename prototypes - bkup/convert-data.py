# import pandas as pd
import dask.dataframe as dd
import shutil


def load_data():
    df = dd.read_csv(
        "/lcrc/project/FastBayes/rayandrew/trace-utils/generated-task/chunk-0.csv"
    )
    return df


df = load_data()
df = df.drop(columns=["instance_name.1"])
df = df.repartition(npartitions=1)
# print(df.head())
df.partitions[0].to_parquet(
    "/lcrc/project/FastBayes/rayandrew/sched-cl/raw_data/alibaba/chunk-0.parquet",
    write_index=False,
)
shutil.move(
    "/lcrc/project/FastBayes/rayandrew/sched-cl/raw_data/alibaba/chunk-0.parquet",
    "/lcrc/project/FastBayes/rayandrew/sched-cl/raw_data/alibaba/tmp",
)
shutil.move(
    "/lcrc/project/FastBayes/rayandrew/sched-cl/raw_data/alibaba/tmp/part.0.parquet",
    "/lcrc/project/FastBayes/rayandrew/sched-cl/raw_data/alibaba/chunk-0.parquet",
)
shutil.rmtree(
    "/lcrc/project/FastBayes/rayandrew/sched-cl/raw_data/alibaba/tmp"
)
