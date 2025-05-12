from typing import List
from pathlib import Path
import shutil

import nannyml as nml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer

N_CHUNKS = 20

non_feature_columns = [
    "name",
    # "task_type",
    "status",
    "start_time",
    "end_time",
    # "instance_num",
    # "plan_cpu",
    # "plan_mem",
    "instance_name",
    "instance_name.1",
    "instance_start_time",
    "instance_end_time",
    "machine_id",
    "seq_no",
    "total_seq_no",
    # "instance_name",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
]


def append_prev_feature(df, num, colname):
    for i in range(1, num + 1):
        df["prev_" + colname + "_" + str(i)] = (
            df[colname].shift(i).values
        )


def create_output_dir(path: str, clean: bool = True) -> Path:
    output_dir = Path(path)

    if clean and output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)

    # output_dir.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


raw_data = pd.read_csv(
    "/lcrc/project/FastBayes/rayandrew/trace-utils/generated-task/chunk-0.csv"
)
raw_data = raw_data.dropna()
raw_data = raw_data[(raw_data.plan_cpu > 0) & (raw_data.plan_mem > 0)]
raw_data = raw_data.sort_values(by=["instance_start_time"])

feature_column_names = [
    col for col in raw_data.columns if col not in non_feature_columns
]


# scaler = StandardScaler()
scaler = Normalizer()
raw_data[feature_column_names] = scaler.fit_transform(
    raw_data[feature_column_names]
)

append_prev_feature(raw_data, 4, "plan_cpu")
append_prev_feature(raw_data, 4, "plan_mem")
append_prev_feature(raw_data, 4, "instance_num")

cpu_avg_pred = pd.cut(
    raw_data.cpu_avg,
    bins=4,
    labels=[0, 1, 2, 3],
)
cpu_max_pred = pd.cut(
    raw_data.cpu_max,
    bins=4,
    labels=[0, 1, 2, 3],
)
mem_avg_pred = pd.cut(
    raw_data.mem_avg,
    bins=4,
    labels=[0, 1, 2, 3],
)
mem_max_pred = pd.cut(
    raw_data.mem_max,
    bins=4,
    labels=[0, 1, 2, 3],
)
raw_data = raw_data.assign(
    cpu_avg_pred=cpu_avg_pred,
    cpu_max_pred=cpu_max_pred,
    mem_avg_pred=mem_avg_pred,
    mem_max_pred=mem_max_pred,
)

feature_column_names += [
    "cpu_avg_pred",
    "cpu_max_pred",
    "mem_avg_pred",
    "mem_max_pred",
]

raw_data = raw_data.dropna()

print(raw_data.head(5))

size = len(raw_data)
split_size = size // N_CHUNKS
subsets: List[pd.DataFrame] = []


for i in range(N_CHUNKS):
    if i == N_CHUNKS - 1:
        data = raw_data.iloc[i * split_size :]
    else:
        data = raw_data.iloc[i * split_size : (i + 1) * split_size]
    subsets.append(data)


def multivariate_drift_calculator():
    calc = nml.DataReconstructionDriftCalculator(
        column_names=feature_column_names,
        # timestamp_column_name="instance_start_time",
        chunk_size=50000,
    )
    calc.fit(subsets[0])

    output_dir = create_output_dir(
        "cov-shift/alibaba/nanny/multivariate"
    )

    for i in range(1, len(subsets)):
        print(f"Processing chunk {i} with len={len(subsets[i])}...")
        results = calc.calculate(subsets[i])
        fig = results.plot()
        fig.write_image(output_dir / f"fig-{i}.png")

        calc = nml.DataReconstructionDriftCalculator(
            column_names=feature_column_names,
            # timestamp_column_name="instance_start_time",
            chunk_size=50000,
        )
        calc.fit(subsets[i])


def univariate_drift_calculator():
    calc = nml.UnivariateDriftCalculator(
        column_names=feature_column_names,
        treat_as_categorical=[
            "cpu_avg_pred",
            "cpu_max_pred",
            "mem_avg_pred",
            "mem_max_pred",
        ],
        continuous_methods=["kolmogorov_smirnov", "jensen_shannon"],
        categorical_methods=["chi2", "jensen_shannon"],
        # timestamp_column_name="instance_start_time",
        chunk_size=250000,
    )
    calc.fit(subsets[0])

    output_dir = create_output_dir(
        "cov-shift/alibaba/nanny/univariate"
    )

    for i in range(1, len(subsets)):
        print(f"Processing chunk {i} with len={len(subsets[i])}...")
        results = calc.calculate(subsets[i])

        fig = results.filter(
            column_names=results.continuous_column_names,
            methods=["jensen_shannon"],
        ).plot(kind="drift")
        fig.write_image(output_dir / f"fig-{i}-js-cont-drift.png")

        fig = results.filter(
            column_names=results.continuous_column_names,
            methods=["jensen_shannon"],
        ).plot(kind="distribution")
        fig.write_image(output_dir / f"fig-{i}-js-cont-dist.png")

        fig = results.filter(
            column_names=results.continuous_column_names,
            methods=["kolmogorov_smirnov"],
        ).plot(kind="drift")
        fig.write_image(output_dir / f"fig-{i}-ks-cont-drift.png")

        fig = results.filter(
            column_names=results.continuous_column_names,
            methods=["kolmogorov_smirnov"],
        ).plot(kind="distribution")
        fig.write_image(output_dir / f"fig-{i}-ks-cont-dist.png")

        fig = results.filter(
            column_names=results.categorical_column_names,
            methods=["chi2"],
        ).plot(kind="drift")
        fig.write_image(output_dir / f"fig-{i}-chi2-cat-drift.png")

        fig = results.filter(
            column_names=results.categorical_column_names,
            methods=["chi2"],
        ).plot(kind="distribution")
        fig.write_image(output_dir / f"fig-{i}-chi2-cat-dist.png")

        fig = results.filter(
            column_names=results.categorical_column_names,
            methods=["jensen_shannon"],
        ).plot(kind="drift")
        fig.write_image(output_dir / f"fig-{i}-js-cat-drift.png")

        fig = results.filter(
            column_names=results.categorical_column_names,
            methods=["jensen_shannon"],
        ).plot(kind="distribution")
        fig.write_image(output_dir / f"fig-{i}-js-cat-dist.png")

        # calc = nml.UnivariateDriftCalculator(
        #     column_names=feature_column_names,
        #     treat_as_categorical=[
        #         "cpu_avg_pred",
        #         "cpu_max_pred",
        #         "mem_avg_pred",
        #         "mem_max_pred",
        #     ],
        #     continuous_methods=[
        #         "kolmogorov_smirnov",
        #         "jensen_shannon",
        #     ],
        #     categorical_methods=["chi2", "jensen_shannon"],
        #     # timestamp_column_name="instance_start_time",
        #     chunk_size=250000,
        # )
        # calc.fit(subsets[i])


# univariate_drift_calculator()
multivariate_drift_calculator()
