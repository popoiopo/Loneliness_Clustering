from functions.metrics import pearson, calc_avg_degree

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import re
import os


def erdos(N, p, e, prefix=0):
    # TODO check if we need to check if these networks are connected
    G = nx.erdos_renyi_graph(N, p, directed=True)
    nx.set_node_attributes(G, {i: {"e": e, "k": e} for i in G.nodes})
    G = nx.relabel_nodes(G, {i: f"{prefix}{i}" for i in G.nodes})
    return G, {e: list(G.edges())}


def barabasi_albert(N, m, e, prefix=0):
    G = nx.barabasi_albert_graph(N, m)
    while not nx.is_connected(G):
        G = nx.barabasi_albert_graph(N, m)

    # Diffuse the tree, otherwise nodes will have scalefree incoming nodes
    # and fixed outgoing
    edges = list(G.edges())
    edges = [
        (edges[i][0], edges[i][1]) if i % 2 == 0 else (edges[i][1], edges[i][0])
        for i in range(len(edges))
    ]

    # Create graph and set attributes
    G = nx.DiGraph()
    G.add_edges_from(edges)
    nx.set_node_attributes(G, {i: {"e": e, "k": e} for i in G.nodes})
    G = nx.relabel_nodes(G, {i: f"{prefix}{i}" for i in G.nodes})
    return G, {e: list(G.edges())}


def holme_kim_graph(N, m, p, e, idx=0):
    G = nx.powerlaw_cluster_graph(N, m, p)
    while not nx.is_connected(G):
        G = nx.barabasi_albert_graph(N, m)
    edges = list(G.edges())
    # Diffuse the tree, otherwise nodes will have scalefree incoming nodes
    # and fixed outgoing
    G = nx.DiGraph()
    G.add_edges_from(edges)
    G = nx.relabel_nodes(G, {i: i + idx for i in G.nodes})
    nx.set_node_attributes(G, {i: {"e": e, "k": e} for i in G.nodes})
    return G, {e: list(G.edges())}


def fully_assortative_network(e_groups, n_per_group, p, network_gen_fn=barabasi_albert):
    G, component_links = network_gen_fn(n_per_group, p, e_groups[0])
    for idx, e in enumerate(e_groups[1:]):
        tmp = G.copy()
        tmp = nx.relabel_nodes(
            tmp, {f"0{i}": f"{idx+1}{i}" for i in range(len(tmp.nodes))}
        )
        nx.set_node_attributes(tmp, {i: {"e": e, "k": e} for i in tmp.nodes})

        G = nx.compose(G, tmp)
        component_links = {**component_links, **{e: list(tmp.edges())}}
    assert pearson(G) == 1, "Network assortativity is fully assortative."
    return G, pd.DataFrame(component_links)


def fully_disassortative_network(
    e_groups, n_per_group, p, network_gen_fn=barabasi_albert
):
    G, component_links = fully_assortative_network(
        e_groups, n_per_group, p, network_gen_fn=network_gen_fn
    )
    initial_ad = calc_avg_degree(G)

    for _, row in component_links.iterrows():
        G.remove_edges_from([row[e_groups[0]], row[e_groups[1]]])
        G.add_edge(row[e_groups[0]][0], row[e_groups[1]][1])
        G.add_edge(row[e_groups[1]][0], row[e_groups[0]][1])

        # Fix component links for later use, it's ugly, I know. But tuples be immutable tuples.
        row[e_groups[0]], row[e_groups[1]] = list(row[e_groups[0]]), list(
            row[e_groups[1]]
        )
        row[e_groups[0]][0], row[e_groups[1]][0] = (
            row[e_groups[1]][0],
            row[e_groups[0]][0],
        )
        row[e_groups[0]], row[e_groups[1]] = tuple(row[e_groups[0]]), tuple(
            row[e_groups[1]]
        )

    # Perform some tests
    assert np.all(
        component_links.apply(
            lambda row: (row[e_groups[0]][0][1:] == row[e_groups[1]][0][1:])
            & (row[e_groups[0]][1][1:] == row[e_groups[1]][1][1:]),
            axis=1,
        )
    ), "Mapping of links incorrect. Check component_links."
    assert pearson(G) == -1.0, "Network assortativity is not -1"
    assert calc_avg_degree(G) == initial_ad, "Average degree changed"
    return G, component_links


def focussed_assort_network_gen(
    aim_assort_value,
    e_groups,
    n_per_group,
    p_rel,
    network_gen_fn=barabasi_albert,
    max_rec=100,
    cur_rec=0,
    debug=False,
):
    if aim_assort_value < 0:
        G, component_links = fully_disassortative_network(
            e_groups, n_per_group, p_rel, network_gen_fn
        )
    else:
        G, component_links = fully_assortative_network(
            e_groups, n_per_group, p_rel, network_gen_fn
        )

    if aim_assort_value == -1.0 or aim_assort_value == 1.0:
        return G, component_links

    # Generate default network
    p = np.round(pearson(G), 2) + 0.0
    avg_degree = calc_avg_degree(G)

    if debug:
        ps = [p]

    rand_component_links = component_links.sample(frac=1)
    for _, row in rand_component_links.iterrows():
        G.remove_edges_from([row[e_groups[0]], row[e_groups[1]]])
        G.add_edge(row[e_groups[0]][0], row[e_groups[1]][1])
        G.add_edge(row[e_groups[1]][0], row[e_groups[0]][1])

        # Calculate pearson and store graph if needed
        p = np.round(pearson(G), 2)
        if debug:
            ps.append(p)

        if p == aim_assort_value:
            if debug:
                print(ps)
                plt.plot(ps)
                plt.show()
            assert (
                calc_avg_degree(G) == avg_degree
            ), f"Average degree changed from {calc_avg_degree(G)} to {avg_degree}, please check rewiring."
            return G, component_links

    if debug:
        print(ps)
        plt.plot(ps)
        plt.show()
    if cur_rec == max_rec:
        raise Exception("Exceeded max recursion.")

    assert (
        calc_avg_degree(G) == avg_degree
    ), f"Average degree changed from {calc_avg_degree(G)} to {avg_degree}, please check rewiring."
    return focussed_assort_network_gen(
        aim_assort_value,
        e_groups,
        n_per_group,
        p_rel,
        network_gen_fn=network_gen_fn,
        cur_rec=cur_rec + 1,
        max_rec=max_rec,
    )


def focussed_assort_networks_gen(
    aim_assort_values,
    e_groups,
    n_per_group,
    p_rel,
    network_gen_fn=barabasi_albert,
    max_rec=100,
    cur_rec=0,
    debug=False,
):
    G_list = []
    p_list = []
    G, component_links = fully_assortative_network(
        e_groups, n_per_group, p_rel, network_gen_fn
    )

    if 1.0 in aim_assort_values:
        G_list.append(G.copy())
        p_list.append(1.0)
        if len(G_list) == len(aim_assort_values):
            return G_list

    # Generate default network
    p = np.round(pearson(G), 2) + 0.0
    avg_degree = calc_avg_degree(G)

    if debug:
        ps = [p]

    rand_component_links = component_links.sample(frac=1)
    for _, row in rand_component_links.iterrows():
        G.remove_edges_from([row[e_groups[0]], row[e_groups[1]]])
        G.add_edge(row[e_groups[0]][0], row[e_groups[1]][1])
        G.add_edge(row[e_groups[1]][0], row[e_groups[0]][1])

        # Calculate pearson and store graph if needed
        p = np.round(pearson(G), 2) + 0.0
        if debug:
            ps.append(p)

        if p in aim_assort_values and p not in p_list:
            if debug:
                print(ps)
                plt.plot(ps)
                plt.show()

            assert (
                calc_avg_degree(G) == avg_degree
            ), f"Average degree changed from {calc_avg_degree(G)} to {avg_degree}, please check rewiring."
            print(f"adding {p=}")
            G_list.append(G.copy())
            p_list.append(p)
            if len(G_list) == len(aim_assort_values):
                return G_list

    if debug:
        print(ps)
        plt.plot(ps)
        plt.show()
    if cur_rec == max_rec:
        raise Exception("Exceeded max recursion.")

    assert (
        calc_avg_degree(G) == avg_degree
    ), f"Average degree changed from {calc_avg_degree(G)} to {avg_degree}, please check rewiring."
    return focussed_assort_networks_gen(
        aim_assort_values,
        e_groups,
        n_per_group,
        p_rel,
        network_gen_fn=network_gen_fn,
        cur_rec=cur_rec + 1,
        max_rec=max_rec,
    )


def write_graph(G, mypath, predefined_name="None"):
    if not os.path.exists(mypath):
        os.makedirs(mypath)

    onlyfiles = [
        f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))
    ]
    filenumbers = [
        int(re.sub("[^0-9]", "", fn))
        for fn in onlyfiles
        if any(i.isdigit() for i in fn)
    ]

    if predefined_name == "None":
        if filenumbers == []:
            fpath = os.path.join(mypath, "0.gml")
        else:
            fpath = os.path.join(mypath, f"{max(filenumbers)+1}.gml")
    else:
        fpath = os.path.join(mypath, predefined_name)

    nx.write_gml(G, fpath)
