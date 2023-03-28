from multiprocessing.pool import ThreadPool as Pool
from functions.network_generation import (
    focussed_assort_network_gen,
    barabasi_albert,
    erdos,
    write_graph,
)

from pathlib import Path

import json
import sys
import os


def main(conf_file, n_networks, output_folder):
    # Get network configuration info
    with open(conf_file, "r") as f:
        conf = json.load(f)
    conf_path = f"{output_folder}/{conf['network_gen_fn']}-{conf['e_samples']}es-{conf['n_per_group']}n-{conf['p_rel']}p"
    if not os.path.exists(conf_path):
        os.makedirs(conf_path)

    def make_networks(a):
        for _ in range(n_networks):
            G, _ = focussed_assort_network_gen(
                a,
                conf["e_samples"],
                conf["n_per_group"],
                conf["p_rel"],
                network_gen_fn=globals().get(conf["network_gen_fn"]),
            )
            write_graph(G, os.path.join(conf_path, str(a)))
        return f"Wrote files for assort {a}."

    pool = Pool(len(conf["a_s"]))
    for result in pool.imap_unordered(make_networks, conf["a_s"]):
        print(result)


if __name__ == "__main__":
    if (args_count := len(sys.argv)) > 4:
        print(f"Two arguments expected, got {args_count - 1}.")
        raise SystemExit(2)
    elif args_count < 3:
        print(
            "You must specify the configuration file, the number of networks you want to generate, and the root output folder."
        )
        raise SystemExit(2)

    # Extract command line arguments for conf file and how many networks
    # there are to be generated.
    conf_file = Path(os.path.join("input/configs/", sys.argv[1]))
    n_networks = int(sys.argv[2])
    output_folder = Path(sys.argv[3])

    if not os.path.exists(conf_file):
        print(f"Given configuration file does not exits: {conf_file}")
        raise SystemExit(2)

    if not os.path.exists(output_folder):
        print(f"Given output folder does not exits: {output_folder}")
        raise SystemExit(2)

    main(conf_file, n_networks, output_folder)
