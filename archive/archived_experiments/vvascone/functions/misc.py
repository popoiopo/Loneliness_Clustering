import numpy as np
import random

def distribute_points_over_cores(cores, exps, points):
    # Divide point simulations over cores
    # If we only look at one point, distribute runs over cores
    # Note: Double list is needed to keep list dimensions same over all cases
    if len(points) == 1:
        distr_points = points
        if cores > exps:
            runs_per_core = 1
            cores = exps
        else:
            runs_per_core = int(exps / cores)
    # If we look at multiple points but they are less than number of cores
    # Distribute points over equal number of cores
    elif cores >= len(points):
        runs_per_core = exps
        cores = len(points)
        distr_points = [[point] for point in points]
    # If we have more points than cores, distribute points over max cores
    else:
        runs_per_core = exps
        distr_points = [[] for _ in range(cores)]
        for index, point in enumerate(points):
            distr_points[index % cores].append(point)
    return cores, runs_per_core, distr_points


def bootstrap_resample_data(data, N):
    bootstrapped_data = [random.choices(data, k=len(data)) for _ in range(N)]
    # print(f"{bootstrapped_data=}")
    # means = []
    # for i in bootstrapped_data:
    #     mean = np.nanmean(i, axis=0)
    #     # print(f"{mean=}")
    #     means.append(mean)
    means = [np.nanmean(i, axis=0) for i in bootstrapped_data]
    mean_of_means = np.mean(means, axis=0)
    # if np.isnan(mean_of_means):
    #     print(f"{bootstrapped_data=}\n{means=}")
    se_of_means = np.std(means, ddof=1, axis=0) / np.sqrt(N)
    return means, mean_of_means, se_of_means