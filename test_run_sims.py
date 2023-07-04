from functions.network_generation import erdos, barabasi_albert, write_graph
from functions.metrics import calc_pearson
from functions.model import mainModel

from multiprocessing.pool import ThreadPool as Pool
import networkx as nx
import numpy as np
import logging
import json
import time
import sys
import re
import os


def write_dyn_data(
    data, data_file_path
):
    # Serializing json
    json_object = json.dumps(data, indent=4)

    # Writing to sample.json
    with open(data_file_path, "w") as outfile:
        outfile.write(json_object)


def check_paths(conf, a):
    base_path = "./testgraphs"
    t0_path = "t0_graphs"
    tt_path = f"tt_graphs"
    dyn_path = f"dyn_data"
    conf_path = f"{conf['network_gen_fn'].__name__}-{conf['e_samples']}es-{conf['n_per_group']}n-{conf['p_rel']}"
    sim_info_path = f"noise_{conf['noise_std']}-b{conf['beta']}-sd{conf['sim_dur']}"

    paths = [
        base_path,
        os.path.join(base_path, conf_path),
        os.path.join(base_path, conf_path, sim_info_path),
        os.path.join(base_path, conf_path, sim_info_path, tt_path),
        os.path.join(base_path, conf_path, sim_info_path, tt_path, str(a)),
        os.path.join(base_path, conf_path, sim_info_path, dyn_path),
        os.path.join(base_path, conf_path, sim_info_path, dyn_path, str(a))
    ]

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    t0_path = os.path.join(base_path, t0_path, conf_path, str(a))
    return t0_path, paths[-3], paths[-1]


def run_sims(a):
    conf = {
        "points": [[float(f) for f in p.split(",")] for p in sys.argv[1:4]],
        "n_per_group": int(sys.argv[4]),
        "noise_std": float(sys.argv[5]),  # 0.1 or 0.02
        "sim_dur": int(sys.argv[6]),
        "e_samples": [0.2, 0.8],
        "n_swaps": 1500,
        "network_gen_fn": barabasi_albert,
        "a": a,
        "beta": 1 / 2,
    }
    conf["window"] = int(conf["sim_dur"] * 0.1)
    if conf["network_gen_fn"] == erdos:
        conf["p_rel"] = 0.2
    elif conf["network_gen_fn"] == barabasi_albert:
        conf["p_rel"] = 11

    t0_path, graph_base_path, dyn_base_path = check_paths(conf, a)

    sim_info = f"Simulation N:{conf['n_per_group']}, noise: {conf['noise_std']}, sim_dur:{conf['sim_dur']} Running points: {conf['points']} for {conf['a']=}"
    logging.info(sim_info)

    files = os.listdir(t0_path)
    for file in files:
        G = nx.read_gml(os.path.join(t0_path, file))

        for point in conf['points']:
            model_path = f"p{'-'.join(str(np.round(p,2)) for p in point)}"

            graph_path = os.path.join(graph_base_path, model_path)

            if not os.path.exists(graph_path):
                os.mkdir(graph_path)

            dyn_data_base_path = os.path.join(dyn_base_path, model_path)

            if not os.path.exists(dyn_data_base_path):
                os.mkdir(dyn_data_base_path)

            dyn_data_path = os.path.join(
                dyn_data_base_path, re.sub("[^0-9]", "", file) + ".json"
            )

            if os.path.exists(dyn_data_path):
                logging.info(
                    f"{sim_info} - File {file} already exists. skipping."
                )
                continue
            logging.info(f"{sim_info} - Running sims for {file} on assort {a}")

            model_parameters = {
                "h": 0.05,
                "noise_std": conf["noise_std"],
                "beta": conf["beta"],
                "point": point,
                "G": G.copy(),
            }
            model = mainModel(**model_parameters)

            # Initialize the datastructures
            es = np.zeros((len(model.G.nodes), conf["sim_dur"]))
            # ks = np.zeros((len(model.G.nodes), sim_dur))
            pearsons = [calc_pearson(model.G)]

            # Run the model
            for t in range(conf["sim_dur"]):
                model.next()

                # Save the data
                for n_idx, node in enumerate(model.G.nodes):
                    es[n_idx, t] = model.G.nodes[node]["e"]
                    # ks[n_idx, t] = model.G.nodes[node]["k"]
                pearsons.append(calc_pearson(model.G))

                # Check if the system has converged
                if t > conf["window"] and np.all(
                    (np.var(es[:, t - conf["window"]: t], axis=1) < 1e-5)
                ):
                    es[:, t:conf["sim_dur"]] = np.tile(
                        es[:, t], (conf["sim_dur"] - t, 1)).T
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
            data = {
                "non_lonely_mean": non_lonely_mean.tolist(),
                "non_lonely_std": non_lonely_std.tolist(),
                "lonely_mean": lonely_mean.tolist(),
                "lonely_std": lonely_std.tolist(),
                "pearsons": pearsons,
            }

            logging.info(
                f"Writing data for assort {a}, file {file}, and for point {point}"
            )
            write_graph(model.G, graph_path, predefined_name=file)
            write_dyn_data(data, dyn_data_path)
    return f"Pool finished for {a}"


if __name__ == "__main__":
    logging.basicConfig(filename="logfile.log",
                        encoding="utf-8", level=logging.DEBUG)
    a_s = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
    tic = time.perf_counter()
    with Pool(processes=len(a_s)) as pool:
        results = pool.imap_unordered(run_sims, a_s)
        for result in results:
            logging.debug(result)

    toc = time.perf_counter()
    logging.info(f"Ran code in {toc - tic:0.4f} seconds")
