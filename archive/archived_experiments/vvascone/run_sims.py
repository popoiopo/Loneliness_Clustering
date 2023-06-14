from functions.network_generation import erdos, barabasi_albert, write_graph
from functions.metrics import calc_pearson
from functions.model import mainModel

from multiprocessing.pool import ThreadPool as Pool
import networkx as nx
import numpy as np
import logging
import json
import time
import re
import os


def write_dyn_data(
    non_lonely_mean, non_lonely_std, lonely_mean, lonely_std, pearsons, data_file_path
):
    data = {
        "non_lonely_mean": non_lonely_mean.tolist(),
        "non_lonely_std": non_lonely_std.tolist(),
        "lonely_mean": lonely_mean.tolist(),
        "lonely_std": lonely_std.tolist(),
        "pearsons": pearsons,
    }

    # Serializing json
    json_object = json.dumps(data, indent=4)

    # Writing to sample.json
    with open(data_file_path, "w") as outfile:
        outfile.write(json_object)


def run_sims(a):
    # points = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # points = [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    points = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    sim_dur = 10000
    window = int(sim_dur * 0.1)
    conf = {
        "e_samples": [0.2, 0.8],
        "n_per_group": 500,
        "n_swaps": 1500,
        "network_gen_fn": barabasi_albert,
        "a": a,
    }
    if conf["network_gen_fn"] == erdos:
        conf["p_rel"] = 0.2
    elif conf["network_gen_fn"] == barabasi_albert:
        conf["p_rel"] = 11

    base_path = "./graphs"
    t0_path = "t0_graphs"
    tt_path = "tt_graphs"
    dyn_path = "dyn_data"
    conf_path = f"{conf['network_gen_fn'].__name__}-{conf['e_samples']}es-{conf['n_per_group']}n-{conf['p_rel']}p"

    paths = [
        os.path.join(base_path, tt_path),
        os.path.join(base_path, tt_path, conf_path),
        os.path.join(base_path, tt_path, conf_path, str(a)),
        os.path.join(base_path, dyn_path),
        os.path.join(base_path, dyn_path, conf_path),
        os.path.join(base_path, dyn_path, conf_path, str(a)),
    ]

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    t0_assort_dir_path = os.path.join(base_path, t0_path, conf_path, str(a))
    files = os.listdir(t0_assort_dir_path)
    for file in files:
        logging.info(f"Running sims for {file} on assort {a}")
        G = nx.read_gml(os.path.join(t0_assort_dir_path, file))

        for point in points:
            model_parameters = {"G": G.copy(), "h": 0.05, "beta": 1 / 2, "point": point}

            model_path = f"p{'-'.join(str(np.round(p,2)) for p in model_parameters['point'])}_b{model_parameters['beta']}_sd{sim_dur}"

            graph_path = os.path.join(base_path, tt_path, conf_path, str(a), model_path)
            if not os.path.exists(graph_path):
                os.mkdir(graph_path)

            dyn_data_path = os.path.join(
                base_path, dyn_path, conf_path, str(a), model_path
            )
            if not os.path.exists(dyn_data_path):
                os.mkdir(dyn_data_path)

            dyn_data_path = os.path.join(
                dyn_data_path, re.sub("[^0-9]", "", file) + ".json"
            )

            if os.path.exists(dyn_data_path):
                logging.info(
                    f"File {file} for point {point} for {a} already exists. skipping."
                )
                continue

            model = mainModel(**model_parameters)

            es = np.zeros((len(model.G.nodes), sim_dur))
            # ks = np.zeros((len(model.G.nodes), sim_dur))
            pearsons = [calc_pearson(model.G)]

            for t in range(sim_dur):
                model.next()
                for n_idx, node in enumerate(model.G.nodes):
                    es[n_idx, t] = model.G.nodes[node]["e"]
                    # ks[n_idx, t] = model.G.nodes[node]["k"]
                pearsons.append(calc_pearson(model.G))

                if t > window and np.all(
                    (np.var(es[:, t - window : t], axis=1) < 1e-5)
                ):
                    es[:, t:sim_dur] = np.tile(es[:, t], (sim_dur - t, 1)).T
                    break

            non_lonely_mean = np.mean(
                es[np.round(es[:, 0], 1) == conf["e_samples"][1]], axis=0
            )
            non_lonely_std = np.std(
                es[np.round(es[:, 0], 1) == conf["e_samples"][1]], axis=0
            )

            lonely_mean = np.mean(
                es[np.round(es[:, 0], 1) == conf["e_samples"][0]], axis=0
            )
            lonely_std = np.std(
                es[np.round(es[:, 0], 1) == conf["e_samples"][0]], axis=0
            )

            logging.info(
                f"Writing data for assort {a}, file {file}, and for point {point}"
            )
            write_graph(model.G, graph_path, predefined_name=file)
            write_dyn_data(
                non_lonely_mean,
                non_lonely_std,
                lonely_mean,
                lonely_std,
                pearsons,
                dyn_data_path,
            )
    return f"Pool finished for {a}"


if __name__ == "__main__":
    logging.basicConfig(filename="logfile.log", encoding="utf-8", level=logging.DEBUG)
    a_s = [0, 0.4, -0.4, 0.8, -0.8, 0.2, -0.2, 0.6, -0.6]
    tic = time.perf_counter()
    with Pool(processes=len(a_s)) as pool:
        results = pool.imap_unordered(run_sims, a_s)
        for result in results:
            logging.debug(result)

    toc = time.perf_counter()
    logging.info(f"Ran code in {toc - tic:0.4f} seconds")
