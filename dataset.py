import math
from pathlib import Path
from typing import Union, Tuple, Literal
import random

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


def split_evenly_by_classes(
    data: pd.DataFrame,
    class_label: str,
    train: bool,
    train_ratio: float,
    shuffle: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data[class_label],
        test_size=1 - train_ratio,
        stratify=data[class_label],
        shuffle=shuffle,
    )

    if train:
        return X_train, y_train

    return X_test, y_test


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
        train_ratio: float = TRAIN_RATIO,
        # train: bool = False,
        y: Literal[
            "cpu_util_percent", "mem_util_percent", "disk_io_percent"
        ] = "cpu_util_percent",
        mode: Literal["train", "test", "predict"] = "train",
    ):
        self.filename = filename
        self.train_ratio = train_ratio
        # self.train = train
        self.n_labels = n_labels
        self.y_var = y
        self.mode = mode
        self.window = 5
        self._load_data()

    @staticmethod
    def get_input_size() -> int:
        return 6 * 5

    def _augment_data(self, datas):
        datas = datas[1:]
        datas = datas[:, 1:]
        ts = datas[:, 0]
        # datas = datas[:, :-1]
        # datas = datas[:, :-1]
        i = 0
        newdatas = []
        # targets = []
        label_index = -2
        if self.y_var == "cpu_util_percent":
            label_index = -4
        elif self.y_var == "mem_util_percent":
            label_index = -3

        while i + self.window <= datas.shape[0]:
            mat = datas[i : i + self.window, :]
            item = mat.flatten()
            data = item[:-3]
            label = item[label_index]
            dist_label = item[-1]
            i += 1
            newdatas.append((data, label, int(dist_label), ts[i]))
            # targets.append(dist_label)
        return newdatas

    def _prepare_targets(self, data):
        targets = []
        for i in range(len(data)):
            # print(data[i])
            targets.append(int(data[i][2]))
        return targets

    def _load_data(self):
        data = np.genfromtxt(self.filename, delimiter=",")
        data[np.isnan(data)] = 0
        new_data = self._augment_data(data)

        train_size = int(len(new_data) * self.train_ratio)
        assert self.mode in ["train", "test", "predict"]
        if self.mode == "train":
            self.data = new_data[:train_size]
            # self.targets = targets[:train_size]
            random.shuffle(self.data)
            self.targets = self._prepare_targets(self.data)
        elif self.mode == "test":
            self.data = new_data[train_size:]
            # self.targets = targets[:train_size]
            random.shuffle(self.data)
            self.targets = self._prepare_targets(self.data)
        else:
            self.data = new_data
            self.targets = self._prepare_targets(self.data)

        # print(self.targets)
        # self.targets = targets

    # def _load_data(self):
    #     self.raw_data = pd.read_csv(self.filename)
    #     self.raw_data = self.raw_data.dropna()
    #     self.raw_data["y"] = self.raw_data[self.y_var]
    #     self.raw_data.loc[:, "y"] = self.raw_data.y.apply(label_transform)
    #     self.raw_data = self.raw_data.reset_index(drop=True)

    #     if self.train_ratio == 1.0 or self.eval:
    #         self.data = self.raw_data
    #         self.y = self.data["y"]
    #     else:
    #         self.data, self.y = split_evenly_by_classes(
    #             self.raw_data,
    #             class_label="y",
    #             train=self.train,
    #             train_ratio=self.train_ratio,
    #         )

    #     self.x = self.data[self.FEATURE_COLUMNS].reset_index(drop=True)
    #     self.y = self.y.reset_index(drop=True)
    #     self.targets = self.data["dist_label"].reset_index(drop=True).to_numpy()

    # def __len__(self):
    #     return len(self.x)

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

    # def __getitem__(self, idx):
    #     if self.eval:
    #         return (
    #             self.x.iloc[idx].astype(np.float32).values,
    #             self.y[idx],
    #             self.data[self.y_var].iloc[idx],
    #             self.data["time_stamp"].iloc[idx],
    #         )
    #     return (
    #         self.x.iloc[idx].astype(np.float32).values,
    #         self.targets[idx],
    #         self.y[idx],
    #     )


__all__ = [
    "AlibabaDataset",
]
