import ruptures as rpt
from src.preprocess.online_cp import Detector, StudentTMulti
from skmultiflow.drift_detection.hddm_w import HDDM_W
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import DDM, KSWIN, PageHinkley
from omegaconf import DictConfig
import argparse
from collections import deque
from pathlib import Path
from typing import Sequence

import hydra
import matplotlib.pyplot as plt
import numpy as np


class DriftDetection:
    def __init__(self, data_stream, verbose=True):
        self.methods = []
        self.weights = []
        self.drifts = []
        self.data_stream = data_stream
        self.verbose = verbose

    def add_method(self, method, weight=1):
        self.methods.append(method)
        self.weights.append(weight)
        self.drifts.append(deque())

    def _get_drift_point(self):
        for pos, ele in enumerate(self.data_stream):
            for method_index, method in enumerate(self.methods):
                method.add_element(ele)
                if method.detected_change():
                    self.drifts[method_index].append(pos)

    def _vote_drift(self, window_size, threshold):
        self.vote_drifts = []
        for i in range(0, len(self.data_stream), window_size):
            pos_sum = 0
            weight_sum = 0
            for method_index in range(len(self.methods)):
                while len(self.drifts[method_index]) != 0:
                    pos = self.drifts[method_index][0]
                    if pos >= (i + 1) * window_size:
                        break
                    else:
                        pos_sum += self.weights[method_index] * pos
                        weight_sum += self.weights[method_index]
                        self.drifts[method_index].popleft()
            if weight_sum != 0 and self.verbose:
                print(weight_sum)
            if weight_sum > threshold:
                mean_pos = int(pos_sum / weight_sum)
                self.vote_drifts.append(mean_pos)

    def get_voted_drift(self, window_size, threshold):
        for method_idx in range(len(self.drifts)):
            self.drifts[method_idx] = deque()
        self._get_drift_point()
        self._vote_drift(window_size, threshold)
        return self.vote_drifts


def plot(stream_window, change, path):
    y = stream_window
    x = [i for i in range(len(y))]

    plt.figure(figsize=(30, 6))

    # Plot the data
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("Stream Data Plot")

    plt.tight_layout()

    # Add vertical lines
    index = [i for i in change]
    for i in index:
        plt.axvline(x=i, color="red", linestyle="--")

    # Show the plot
    plt.savefig(f"{path}")


def add_dist_label(data, dist: Sequence[int], start_from=0):
    distributions = np.zeros((len(data[:, 0]), 1))

    print(dist)
    print(f"Dist from 0 to {dist[0]}: 0")
    distributions[: dist[0]] = start_from
    for i in range(len(dist) - 2):
        print(f"Dist from {dist[i]} to {dist[i+1]}: {i+1}")
        distributions[dist[i]: dist[i + 1]] = i + 1 + start_from
    print(
        f"Dist from {dist[len(dist) - 1]} to {dist[-1]}: {len(dist) - 1}"
    )
    distributions[dist[-1]:] = len(dist) - 1 + start_from

    print(np.unique(distributions, return_counts=True))

    data = np.append(data, distributions, axis=1)
    return data


@hydra.main(
    config_path="../../config",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    input_path = Path(cfg.dataset.raw_path)
    orig_data = np.genfromtxt(input_path, delimiter=",")

    outfile_path = Path(cfg.filename)
    output_path = outfile_path.parent
    output_path.mkdir(parents=True, exist_ok=True)

    data = orig_data

    if "alibaba" in cfg.dataset.name:
        if cfg.local:
            label_index = 9
            if cfg.y == "cpu":
                label_index = 7
            elif cfg.y == "mem":
                label_index = 8
            orig_data = orig_data[1:]
            data = data[1:]
        else:
            label_index = 8
            if cfg.y == "cpu":
                label_index = 2
            elif cfg.y == "mem":
                label_index = 3
    elif "google" in cfg.dataset.name:
        data = data[1:]  # google contains csv headers
        if cfg.y == "cpu":
            label_index = 2
        elif cfg.y == "mem":
            label_index = 3
        else:
            raise ValueError("Invalid y value")

    # Nx2 Matrix, with columns end time and cpu usage.
    data_end = data[:, 1:3]
    data = data[:, label_index]
    window_size = cfg.drift.window_size
    threshold = cfg.drift.threshold

    change_list = []
    dd = DriftDetection(data, verbose=False)
    if "alibaba" in cfg.dataset.name:
        print(
            f"Start detecting drifts with window_size {window_size} and threshold {threshold}"
        )
        dd.add_method(ADWIN(), 1)
        dd.add_method(DDM(), 1)
        dd.add_method(EDDM(), 1)
        dd.add_method(HDDM_A(), 1)
        dd.add_method(HDDM_W(), 1)
        dd.add_method(PageHinkley(), 1)
        dd.add_method(KSWIN(), 1)
    elif "google" in cfg.dataset.name:
        method = cfg.drift.google_method
        # can put this in dvc
        if "ruptures" in method:
            # Ruptures model. It performs really well!
            model = rpt.Pelt(model="rbf").fit(data)

            # get the change point locations higher penalty = less change points. The model won't want to predict wrong.
            change_list = model.predict(pen=cfg.drift.ruptures_penalty)[:-1]
        # CAUTION! Takes a long time
        elif "online_cp" in method:

            # print the change point locations

            # data_end = Nx2 Matrix, with columns end time and cpu usage.
            # Number of rows
            detector = Detector(data_end.shape[0])
            # Number of cols
            observation_likelihood = StudentTMulti(data_end.shape[1])

            R_mat = np.zeros((data_end.shape[0], data_end.shape[0]))
            R_mat_cumfreq = np.zeros((data_end.shape[0], data_end.shape[0]))
            R_mat.fill(np.nan)

            CP = np.array([])
            for t, x in enumerate(data_end[:, :]):
                # ## Attempt1: Normalize end_time because if not, value will be too big and matrix will be singular.
                # x[0] = x[0]-data_end[0][0]
                print(t, x[0])
                # Attemp2: Use 1...N instead of end_time because end_time has regular 5 minutes interval anyways.
                x[0] = t
                print(t)
                # print(CP)
                detector.detect(
                    x, observation_likelihood=observation_likelihood
                )
                _, CP, _, _ = detector.retrieve(observation_likelihood)

                R_old = detector.R_old

                try:
                    R_mat[t, 0: len(R_old)] = R_old
                    R_mat_cumfreq[t, 0: len(R_old)] = np.cumsum(R_old)
                except:
                    R_mat[t, 0: len(R_old)] = R_old[0:-1]
                    R_mat_cumfreq[t, 0: len(R_old)] = np.cumsum(
                        R_old[0:-1]
                    )
            # print(CP)
            # change_list = CP

            # R_mat2 = R_mat.copy()
            R_mat = R_mat.T

            R_mat_cumfreq = R_mat_cumfreq.T
            # R_mat_median = np.nanmedian(R_mat_cumfreq, axis=1)

            T = R_mat.shape[1]
            Mrun = np.zeros(T)

            for ii in range(T):
                try:
                    Mrun[ii] = np.where(R_mat_cumfreq[:, ii] >= 0.5)[0][0]
                except:
                    pass

            # MchangeTime = np.asarray(range(T)) - Mrun + 1

            #########################################################################
            # Find the max value in Mrun sequentially
            # Check if the next value dropped a certain relative value
            # Check if that drop sustains for 10 points
            CP_CDF = [0]
            for i in range(len(Mrun) - 5):
                j = i + 1
                if (Mrun[i] - Mrun[j]) > 5:
                    cnt = 0
                    for k in range(1, 20):
                        if (i + k < len(Mrun)) and ((Mrun[i] - Mrun[i + k]) > 10):
                            cnt = cnt + 1
                        else:
                            break
                    if cnt > 10:
                        CP_CDF.append(i + 1)
            print("CP_CDF")
            print(CP_CDF)
            change_list = CP_CDF

    # dd = KSWIN(window_size=window_size)
    # for pos, ele in enumerate(data):
    #     dd.add_element(ele)
    #     if dd.detected_change():
    #         change_list.append(pos)

    # change_list = dd.get_voted_drift(
    #     window_size=window_size, threshold=threshold
    # )
    # change_list = [i + 1000 for i in change_list]
    change_list = sorted(change_list)
    # rec_time = int(time.time())

    n_dist = len(change_list)
    if n_dist == 0:
        print("No change detected")
        # return

    print(f"Number of distributions: {n_dist}")
    plot(
        data,
        change_list,
        output_path
        / f"plot_{input_path.stem}_{window_size}_{threshold}_{cfg.y}.png",
    )

    data = add_dist_label(orig_data, change_list, start_from=0)
    np.savetxt(
        # output_path / f"{input_path.stem}_{cfg.y}.csv",
        outfile_path,
        data,
        fmt="%.4e",
        delimiter=",",
    )

    changes = np.zeros((len(change_list), 2))
    changes[:, 0] = change_list

    for i in range(len(change_list)):
        changes[i, 1] = orig_data[
            change_list[i] - 1000, 1
        ]  # get timestamp

    np.savetxt(
        output_path / f"{input_path.stem}_{cfg.y}_cp.csv",
        changes,
        fmt="%d",
        delimiter=",",
    )

    # print(data[0:10, :])
    # print(data[-10:, :])


if __name__ == "__main__":
    main()
#     parser = argparse.ArgumentParser(
#         description="[Preprocess] Drift Detection"
#     )
#     parser.add_argument(
#         "input",
#         type=str,
#     )
#     parser.add_argument(
#         "-o",
#         "--output",
#         type=str,
#         default=f"preprocessed_data",
#     )
#     parser.add_argument(
#         "-y",
#         type=str,
#         choices=["cpu", "mem", "disk"],
#         default="cpu",
#         help="choose the y axis",
#     )
#     parser.add_argument(
#         "--threshold",
#         type=int,
#         default=300,
#         help="threshold for voting",
#     )
#     parser.add_argument(
#         "--window_size",
#         type=int,
#         default=75,
#         help="window size for voting",
#     )
#     parser.add_argument(
#         "--local",
#         action="store_true",
#     )

#     cfg = parser.parse_cfg()
#     main(cfg)

__all__ = ["DriftDetection", "add_dist_label"]
