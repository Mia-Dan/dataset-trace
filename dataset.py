from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Union, Tuple, Literal, Sequence
import random
from numbers import Number

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import numpy.typing as npt


def split_evenly_by_classes(
    X: Sequence[Tuple[npt.ArrayLike, Number, Number]],
    y: Sequence[Number],
    train_ratio: float,
    shuffle: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=1 - train_ratio,
        stratify=y,
        shuffle=shuffle,
    )

    return X_train, X_test, y_train, y_test


def label_transform(x: float | int, n_labels: int = 10) -> int:
    # we know that the range spans between 0.0 and 100.0
    # we can divide them equally right away based on `n_labels`

    if x <= 0.0:  # edge cases
        return 0

    if x >= 100.0:
        return n_labels - 1

    divider = 100.0 / n_labels

    return min(math.ceil(x / divider) - 1, n_labels - 1)


class AlibabaDataset(Dataset):
    TRAIN_RATIO = 0.8

    @abstractmethod
    def input_size(self) -> int:
        raise NotImplementedError

    def _process_nan(self, arr, fill=0):
        # change to previous row
        mask = np.isnan(arr[0])
        arr[0][mask] = fill
        for i in range(1, len(arr)):
            mask = np.isnan(arr[i])
            arr[i][mask] = arr[i - 1][mask]
        return arr


class AlibabaSchedulerDataset(AlibabaDataset):
    FEATURE_COLUMNS = [
        # "time_stamp",
        "cpu_avg",
        "cpu_max",
        "mem_avg",
        "mem_max",
        "plan_cpu",
        "plan_mem",
        # "cpu_util_percent",
        # "mem_util_percent",
        # "disk_io_percent"
    ]

    def __init__(
        self,
        filename: Union[str, Path],
        n_labels: int,
        train_ratio: float = AlibabaDataset.TRAIN_RATIO,
        y: Literal["cpu", "mem", "disk"] = "cpu",
        mode: Literal["train", "test", "predict"] = "train",
    ):
        self.filename = filename
        self.train_ratio = train_ratio
        self.n_labels = n_labels
        self.y_var = y
        self.mode = mode
        self.seq_len = 1
        self._load_data()

    def _augment_data(self, data):
        data = data[1:]
        data = data[:, 1:]
        ts = data[:, 0]
        new_data = []
        label_index = -2
        if self.y_var == "cpu":
            label_index = -4
        elif self.y_var == "mem":
            label_index = -3

        if self.seq_len > 1:
            i = 0
            while i + self.seq_len <= data.shape[0]:
                mat = data[i : i + self.seq_len, :]
                item = mat.flatten()
                data = item[:-3]
                label = item[label_index]
                dist_label = item[-1]
                i += 1
                new_data.append((data, label, int(dist_label), ts[i]))
        else:
            for i, data in enumerate(data):
                item = data.flatten()
                data = item[:-3]
                label = item[label_index]
                dist_label = item[-1]
                new_data.append((data, label, int(dist_label), ts[i]))
        return new_data

    def _prepare_targets(self, data):
        targets = []
        for i in range(len(data)):
            targets.append(int(data[i][2]))
        return targets

    def input_size(self) -> int:
        if self.data is None:
            raise ValueError("Dataset not loaded yet")
        if len(self.data) == 0:
            raise ValueError("Dataset is empty")
        return len(self.data[0][0])

    def _load_data(self):
        data = np.genfromtxt(self.filename, delimiter=",")
        data = self._process_nan(data)
        new_data = self._augment_data(data)

        train_size = int(len(new_data) * self.train_ratio)
        assert self.mode in ["train", "test", "predict"]
        if self.mode == "train":
            self.data = new_data[:train_size]
            random.shuffle(self.data)
            self.targets = self._prepare_targets(self.data)
        elif self.mode == "test":
            self.data = new_data[train_size:]
            random.shuffle(self.data)
            self.targets = self._prepare_targets(self.data)
        else:
            self.data = new_data
            self.targets = self._prepare_targets(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index][0]
        label = np.array(int(min(self.data[index][1], 99) / 10))
        dist_label = self.data[index][2]
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.from_numpy(label)
        dist_label_tensor = torch.from_numpy(np.array(dist_label))
        return data_tensor, dist_label_tensor, label_tensor


class AlibabaMachineDataset(AlibabaDataset):
    CACHE = {}

    def __init__(
        self,
        filename: Union[str, Path],
        n_labels: int,
        train_ratio: float = AlibabaDataset.TRAIN_RATIO,
        y: Literal["cpu", "mem", "disk"] = "cpu",
        mode: Literal["train", "test", "predict"] = "train",
        seq: bool = False,
        seq_len: int = 2,
    ):
        """Dataset for Alibaba Machine dataset

        Args:
            filename (Union[str, Path]): Path to the dataset file
            n_labels (int): Number of labels to use
            train_ratio (float, optional): Ratio of training data. Defaults to AlibabaDataset.TRAIN_RATIO.
            y (Literal["cpu", "mem", "disk"], optional): Variable to predict. Defaults to "cpu".
            mode (Literal["train", "test", "predict"], optional): Mode of the dataset. Defaults to "train".
            seq (bool, optional): Whether to use sequence data. Defaults to False.
            seq_len (int, optional): Sequence length for sequence data. Defaults to 2.
        """
        assert mode in ["train", "test", "predict"]
        assert y in ["cpu", "mem", "disk"]
        assert seq_len >= 2
        self.filename = filename
        self.train_ratio = train_ratio
        self.n_labels = n_labels
        self.y_var = y
        self.mode = mode
        self.seq = seq
        self.seq_len = seq_len
        self._load_data()

    def _augment_data(self, data):
        # do not need code below as the data should come without header
        # data = data[1:]
        # ts = data[:, 1] # timestamp
        # TODO: this is hacky solution, need to fix
        # X need to accomodate "data" and dist_labels together
        # such that we can use `train_test_split` to split the data
        # in the future, we should not use these two variables together
        Xs = []
        Ys = []
        # dist_labels = []

        label_index = 8
        if self.y_var == "cpu":
            label_index = 2
        elif self.y_var == "mem":
            label_index = 3

        dist_labels = data[:, -1]
        # print(dist_labels, np.unique(dist_labels))
        labels = data[:, label_index]
        data = np.delete(data, label_index, axis=1)
        data = data[:, 2:-1]  # remove machine id + timestamp + dist_label
        # print(dist_labels, np.unique(dist_labels))

        if self.seq:
            i = 0
            while i + self.seq_len <= data.shape[0]:
                mat = data[i : i + self.seq_len, :]
                x = mat.flatten()
                ys = labels[i : i + self.seq_len]
                x = np.append(x, ys[:-1])
                y = ys[-1]
                i += 1
                dist_label = dist_labels[i]
                Xs.append((x, dist_label))
                Ys.append(y)
                # new_data.append((x, y, int(dist_label)))
        else:
            for i, d in enumerate(data):
                x = d.flatten()
                y = labels[i]
                dist_label = dist_labels[i]
                Xs.append((x, dist_label))
                Ys.append(y)
                # new_data.append((x, y, int(dist_label)))

        return Xs, Ys

    def input_size(self) -> int:
        if self.data is None:
            raise ValueError("Dataset not loaded yet")
        if len(self.data) == 0:
            raise ValueError("Dataset is empty")
        return len(self.data[0])

    def _load_data(self):
        assert self.mode in ["train", "test", "predict"]

        key = (self.filename, self.train_ratio, self.mode)
        if key in self.CACHE:
            self.data, self.targets, self.outputs = self.CACHE[key]

        data = np.genfromtxt(self.filename, delimiter=",")
        data = self._process_nan(data)
        X_dirty, y = self._augment_data(data)

        if self.mode == "predict":
            X = [x[0] for x in X_dirty]
            dist_labels = [x[1] for x in X_dirty]
            self.data = X
            self.targets = dist_labels
            self.outputs = y
        else:
            X_dirty_train, X_dirty_test, y_train, y_test = split_evenly_by_classes(
                X_dirty, y, train_ratio=self.train_ratio
            )

            X_train = [x[0] for x in X_dirty_train]
            dist_labels_train = [int(x[1]) for x in X_dirty_train]

            X_test = [x[0] for x in X_dirty_test]
            dist_labels_test = [int(x[1]) for x in X_dirty_test]

            self.CACHE[(self.filename, self.train_ratio, "train")] = (
                X_train,
                dist_labels_train,
                y_train,
            )
            self.CACHE[(self.filename, self.train_ratio, "test")] = (
                X_test,
                dist_labels_test,
                y_test,
            )

            if self.mode == "train":
                self.data = X_train
                self.targets = dist_labels_train
                self.outputs = y_train
                # self.data = [x[0] for x in X_train]  # data (X)
                # self.targets = [x[1] for x in X_train]  # dist label
                # self.outputs = y_train
            elif self.mode == "test":
                self.data = X_test
                self.targets = dist_labels_test
                self.outputs = y_test
                # self.data = [x[0] for x in X_test]
                # self.targets = [x[1] for x in X_test]
                # self.outputs = y_test

        # print("train size", len(self.data), "targets size", len(self.targets))
        # print("unique", np.unique(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = np.array(int(min(self.outputs[index], 99) / self.n_labels))
        dist_label = self.targets[index]
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.from_numpy(label)
        dist_label_tensor = torch.from_numpy(np.array(dist_label))
        return data_tensor, dist_label_tensor, label_tensor


__all__ = ["AlibabaSchedulerDataset", "AlibabaMachineDataset"]

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # parser.add_argument("-d", "--data", type=str, default="data/mu_dist/m_25.csv")
    parser.add_argument(
        "-d", "--data", type=str, default="out_preprocess/m_25/m_25.csv"
    )
    parser.add_argument("-n", "--n_labels", type=int, default=10)
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["train", "test", "predict"],
        default="train",
    )
    parser.add_argument(
        "-y",
        type=str,
        choices=["cpu", "mem", "disk"],
        default="cpu",
    )
    parser.add_argument(
        "-s",
        "--seq",
        action="store_true",
    )
    parser.add_argument(
        "-w",
        "--seq_len",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    dataset = AlibabaMachineDataset(
        filename=args.data,
        n_labels=args.n_labels,
        y=args.y,
        seq=args.seq,
        seq_len=args.seq_len,
        mode=args.mode,
    )
    print("INPUT SIZE", dataset.input_size())
    for d in dataset:
        print(d)
        break
