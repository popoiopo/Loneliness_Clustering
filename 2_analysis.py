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


def main(conf_file, input_folder):
    # Get network configuration info
    with open(conf_file, "r") as f:
        conf = json.load(f)
    conf_path = f"{input_folder}/{conf['network_gen_fn']}-{conf['e_samples']}es-{conf['n_per_group']}n-{conf['p_rel']}p"
    if not os.path.exists(conf_path):
        raise


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
    output_folder = Path(sys.argv[3])

    if not os.path.exists(conf_file):
        print(f"Given configuration file does not exits: {conf_file}")
        raise SystemExit(2)

    if not os.path.exists(output_folder):
        print(f"Given output folder does not exits: {output_folder}")
        raise SystemExit(2)

    main(conf_file, output_folder)
