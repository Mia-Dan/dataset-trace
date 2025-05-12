import shutil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler

N_CHUNKS = 20
# COLORS = ["red", "blue"]
# COLORS = ["red", "blue", "green", "yellow", "black", "orange", "purple", "pink", "brown", "gray"]
COLORS = sns.color_palette("hls", N_CHUNKS)

COL_1 = 4  # prev_plan_cpu_1
COL_1_STR = "prev_plan_cpu_1"
COL_2 = 8  # prev_plan_mem_1
COL_2_STR = "prev_plan_mem_1"

# COL_2 = 12  # prev_instance_num_1
# COL_2_STR = "prev_instance_num_1"

non_feature_columns = [
    "name",
    "status",
    "start_time",
    "end_time",
    # "task_type",
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


def create_output_dir(path: str | Path, clean: bool = True) -> Path:
    output_dir = Path(path)

    if clean and output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)

    # output_dir.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def append_prev_feature(df, num, colname):
    for i in range(1, num + 1):
        df["prev_" + colname + "_" + str(i)] = (
            df[colname].shift(i).values
        )


def load_raw_data(path: Path | str):
    raw_data = pd.read_csv(path)

    raw_data = (
        raw_data.sort_values(by=["instance_start_time"])
        .drop(columns=non_feature_columns)
        .dropna()
    )

    raw_data = raw_data[
        (raw_data.plan_cpu > 0) & (raw_data.plan_mem > 0)
    ]

    append_prev_feature(raw_data, 4, "plan_cpu")
    append_prev_feature(raw_data, 4, "plan_mem")
    append_prev_feature(raw_data, 4, "instance_num")

    raw_data = raw_data.dropna()

    # scaler = StandardScaler()
    scaler = Normalizer()
    raw_data[raw_data.columns] = scaler.fit_transform(
        raw_data[raw_data.columns]
    )
    return raw_data


def generate_data_chunks(
    path: Path | str, out_path: Path | str
) -> List[pd.DataFrame]:
    path = Path(path)
    out_path = create_output_dir(out_path, clean=False)

    subsets: List[pd.DataFrame] = []
    for i in range(N_CHUNKS):
        if (out_path / f"chunk-{i}.csv").exists():
            print(f"Chunk {i} already exists, skip generating...")
            subsets.append(pd.read_csv(out_path / f"chunk-{i}.csv"))

    if len(subsets) == N_CHUNKS:
        return subsets

    print("Generating data chunks...")

    raw_data = load_raw_data(path)

    size = len(raw_data)
    split_size = size // N_CHUNKS

    for i in range(N_CHUNKS):
        if i == N_CHUNKS - 1:
            data = raw_data.iloc[i * split_size :]
        else:
            data = raw_data.iloc[
                i * split_size : (i + 1) * split_size
            ]
        data.to_csv(out_path / f"chunk-{i}.csv", index=False)
        subsets.append(data)

    return subsets


def get_pca_series(pca: PCA, data: pd.DataFrame):
    indices = np.argsort(np.abs(pca.components_[0]))[::-1]
    names = [data.columns[i] for i in indices]
    pca_series = pd.Series(
        np.abs(pca.components_[0][indices]), index=names
    )
    return pca_series


def plot_pca_0_to_all(subsets: List[pd.DataFrame]):
    single_pca_output_dir = create_output_dir(
        "./cov-shift/alibaba/single-pca"
    )
    pca = PCA(n_components=2)
    pca.fit(subsets[0])
    pca_series = get_pca_series(pca, subsets[0])

    for i in range(N_CHUNKS):
        # gridspec
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])

        # ax0.scatter(
        #     subsets[i].iloc[:, COL_1],
        #     subsets[i].iloc[:, COL_2],
        #     color="grey",
        #     label=f"Data Chunk {i}",
        #     alpha=0.3,
        # )
        # ax0.scatter(
        #     subsets[i - 1].iloc[:, COL_1],
        #     subsets[i - 1].iloc[:, COL_2],
        #     color="black",
        #     label=f"Data Chunk {i-1}",
        #     alpha=0.8,
        # )

        ax0.scatter(
            subsets[i - 1].iloc[:, COL_1],
            subsets[i - 1].iloc[:, COL_2],
            label=f"Data Chunk {i-1}",
            color="grey",
            alpha=0.3,
        )
        ax0.scatter(
            subsets[i].iloc[:, COL_1],
            subsets[i].iloc[:, COL_2],
            label=f"Data Chunk {i}",
            color="black",
        )

        res = pca.transform(subsets[i])
        subset_from_pca = pca.inverse_transform(res)
        rec_loss_0 = np.linalg.norm(subsets[i] - subset_from_pca)
        # ax0.scatter(
        #     subset_from_pca[:, COL_1],
        #     subset_from_pca[:, COL_2],
        #     color=COLORS[0],
        #     label=f"PCA of CHUNK-0 ({rec_loss_0})",
        #     alpha=0.5,
        # )

        curr_pca = PCA(n_components=2)
        curr_res = curr_pca.fit_transform(subsets[i])
        subset_from_curr_pca = curr_pca.inverse_transform(curr_res)
        rec_loss_1 = np.linalg.norm(subsets[i] - subset_from_curr_pca)
        # ax0.scatter(
        #     subset_from_curr_pca[:, COL_1],
        #     subset_from_curr_pca[:, COL_2],
        #     color=COLORS[1],
        #     label=f"PCA of CHUNK-{i} ({rec_loss_1})",
        #     alpha=0.5,
        # )
        ax0.legend(loc="upper right")
        ax0.set_xlabel(COL_1_STR)
        ax0.set_ylabel(COL_2_STR)

        pca_series.plot.bar(
            y=np.abs(pca.components_[0]),
            ax=ax1,
            color=COLORS[0],
            alpha=0.5,
            label="C-0 Variables Importances",
        )
        ax1.set_title("Variables")
        ax1.set_ylabel("PCA Variables Importances")
        ax1.set_xticklabels(
            ax1.get_xticklabels(), rotation=45, ha="right"
        )
        # ax1.tick_params(axis="x", labelrotation=45, ha="right")
        ax1.legend(loc="upper right")

        curr_pca_series = get_pca_series(curr_pca, subsets[i])
        curr_pca_series.plot.bar(
            y=np.abs(curr_pca.components_[0]),
            ax=ax2,
            color=COLORS[1],
            alpha=0.5,
            label=f"C-{i} Variables Importances",
        )
        ax2.set_title("Variables")
        ax2.set_ylabel("PCA Variables Importances")
        ax2.set_xticklabels(
            ax2.get_xticklabels(), rotation=45, ha="right"
        )
        ax2.legend(loc="upper right")

        fig.suptitle(
            f"PCA from CHUNK-0 to CHUNK-{i}, loss abs diff: {abs(rec_loss_0 - rec_loss_1)}"
        )
        fig.tight_layout()
        print(f"Saving Single PCA fig_{i}.png")
        fig.savefig(single_pca_output_dir / f"fig_{i}.png")
        plt.close(fig)


def plot_pca_prev_to_next(subsets: List[pd.DataFrame]):
    print("Plotting PCA")
    pca_output_dir = create_output_dir("./cov-shift/alibaba/pca")
    pca = PCA(n_components=2)
    pca.fit(subsets[0])

    for i in range(1, N_CHUNKS):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])

        ax0.scatter(
            subsets[i - 1].iloc[:, COL_1],
            subsets[i - 1].iloc[:, COL_2],
            label=f"Data Chunk {i-1}",
            color="grey",
            alpha=0.3,
        )
        ax0.scatter(
            subsets[i].iloc[:, COL_1],
            subsets[i].iloc[:, COL_2],
            label=f"Data Chunk {i}",
            color="black",
        )

        res = pca.transform(subsets[i])
        subset_from_pca = pca.inverse_transform(res)
        rec_loss_0 = np.linalg.norm(subsets[i] - subset_from_pca)
        # ax0.scatter(
        #     subset_from_pca[:, COL_1],
        #     subset_from_pca[:, COL_2],
        #     color=COLORS[0],
        #     label=f"PCA of CHUNK-{i-1} ({rec_loss_0})",
        #     alpha=0.5,
        # )

        curr_pca = PCA(n_components=2)
        curr_res = curr_pca.fit_transform(subsets[i])
        subset_from_curr_pca = curr_pca.inverse_transform(curr_res)
        rec_loss_1 = np.linalg.norm(subsets[i] - subset_from_curr_pca)
        # ax0.scatter(
        #     subset_from_curr_pca[:, COL_1],
        #     subset_from_curr_pca[:, COL_2],
        #     color=COLORS[1],
        #     label=f"PCA of CHUNK-{i} ({rec_loss_1})",
        #     alpha=0.5,
        # )
        ax0.legend(loc="upper right")
        ax0.set_xlabel(COL_1_STR)
        ax0.set_ylabel(COL_2_STR)

        pca_series = get_pca_series(pca, subsets[i])
        pca_series.plot.bar(
            y=np.abs(pca.components_[0]),
            ax=ax1,
            color=COLORS[0],
            alpha=0.5,
            label=f"C-{i-1} Variables Importances",
        )
        ax1.set_title("Variables")
        ax1.set_ylabel("PCA Variables Importances")
        ax1.set_xticklabels(
            ax1.get_xticklabels(), rotation=45, ha="right"
        )
        ax1.legend(loc="upper right")

        curr_pca_series = get_pca_series(curr_pca, subsets[i])
        curr_pca_series.plot.bar(
            y=np.abs(curr_pca.components_[0]),
            ax=ax2,
            color=COLORS[1],
            alpha=0.5,
            label=f"C-{i} Variables Importances",
        )
        ax2.set_title("Variables")
        ax2.set_ylabel("PCA Variables Importances")
        ax2.set_xticklabels(
            ax2.get_xticklabels(), rotation=45, ha="right"
        )
        ax2.legend(loc="upper right")

        pca = PCA(n_components=2)
        pca.fit(subsets[i])

        print(f"Saving PCA fig_{i}.png")
        fig.suptitle(
            f"PCA from CHUNK-{i-1} and CHUNK-{i} to CHUNK-{i}, loss abs diff: {abs(rec_loss_0 - rec_loss_1)}"
        )
        fig.tight_layout()
        fig.savefig(pca_output_dir / f"fig_{i}.png")
        plt.close(fig)


def plot_density(subsets: List[pd.DataFrame]):
    density_output_dir = create_output_dir(
        f"./cov-shift/alibaba/density"
    )

    for column in subsets[0].columns:
        fig, ax = plt.subplots(figsize=(15, 10))
        for i in range(N_CHUNKS - 1):
            sns.kdeplot(
                data=subsets[i],
                x=column,
                ax=ax,
                color=COLORS[i],
                alpha=0.5,
                bw=0.5,
                label=f"CHUNK-{i}",
            )

        print(f"Saving {column} Density")
        ax.legend(loc="upper right")
        fig.suptitle(f"{column} Density")
        fig.tight_layout()
        fig.savefig(density_output_dir / f"{column}.png", dpi=300)
        plt.close(fig)


def plot_cdf(subsets: List[pd.DataFrame]):
    cdf_output_dir = create_output_dir(f"./cov-shift/alibaba/cdf")
    for column in subsets[0].columns:
        fig, ax = plt.subplots(figsize=(15, 10))
        for i in range(N_CHUNKS):
            sns.ecdfplot(
                data=subsets[i],
                x=column,
                ax=ax,
                color=COLORS[i],
                alpha=0.5,
                label=f"CHUNK-{i}",
            )
            # sns.ecdfplot(
            #     data=subsets[i + 1],
            #     x=column,
            #     ax=ax,
            #     color=COLORS[1],
            #     alpha=0.5,
            #     label=f"CHUNK-{i + 1}",
            # )
        print(f"Saving {column} CDF")
        ax.legend(loc="upper right")
        fig.suptitle(f"{column} CDF")
        fig.tight_layout()
        fig.savefig(cdf_output_dir / f"{column}.png", dpi=300)
        plt.close(fig)


def main() -> None:
    subsets = generate_data_chunks(
        "/lcrc/project/FastBayes/rayandrew/trace-utils/generated-task/chunk-0.csv",
        "./cov-shift/alibaba/chunks-norm",
    )
    print(subsets[0].columns)
    # prev_plan_cpu_1
    # prev_instance_num_1

    # plot_pca_0_to_all(subsets)
    # plot_pca_prev_to_next(subsets)
    # plot_cdf(subsets)
    plot_density(subsets)


main()
