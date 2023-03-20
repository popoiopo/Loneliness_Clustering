import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import re
from itertools import chain
import os
from os import listdir
from os.path import isfile, join

from functions.metrics import pearson, calc_avg_degree
import researchhelper.visualize.general_formatting as gf


def create_network(N, groups, group_ratio, group_variable, p, network_type):
    """Creates a bipartite graph of size N divided by different groups. The groups are divided by a group ratio.

    Args:
        N (int): Size of the total network.
        groups (list):  List of dicts with parameter values for each group, each dict has to have at least one
                        unique key value pair per group.
        group_ratio (list): List of ratio's for each groups representation in the network. Must sum up to 1.
        group_variable (dict key): Dictionary key of the groups parameter on which groups are formed.
        p (float): Probability of creating a link.
        network_type (operator.eq or operator.ne):  Variable to check if nodes connect to same group, assortativity == 1 (input == operator.eq)
                                                    or that nodes connect only to other group, assortativity == -1 (input == operator.ne)
                                                    or an erdos renyi network, assortativity == 0 (input == lambda x,y: True)
    """
    # Calculate number of agents per group
    cumsum = np.concatenate(
        (np.array([0]), np.cumsum(list(map(int, np.array(group_ratio) * N)))))
    assert cumsum[-1] == N, f"length not the same: {cumsum[-1]=} vs. {N=}. Ratio's ({group_ratio}) incompatible with number of nodes."

    # Create graph and populate with agents
    graph = nx.DiGraph()
    for i, group in enumerate(groups):
        graph.add_nodes_from([(agent, group)
                              for agent in range(cumsum[i], cumsum[i+1])])

    # Create relationships between agents based on the group belonging.
    edges = np.array([(i, j, {"weight": 1}) for i in graph.nodes(
    ) for j in graph.nodes()
        if network_type(graph.nodes[i][group_variable], graph.nodes[j][group_variable]) and
        (np.random.random() < p)])
    graph.add_edges_from(edges)

    return graph


def swap_edge(G, component_links):
    sG = G.copy()

    # Check which clusters are not empty
    ckeys = [k for k in component_links if component_links[k] != []]

    # Check if there are any clusters left to swap between
    if len(ckeys) < 2:
        return sG.copy(), component_links

    # Choose two relationships
    c1, c2 = random.sample(ckeys, 2)
    if component_links[c1] != [] and component_links[c2] != []:
        ec1, ec2 = random.choice(
            component_links[c1]), random.choice(component_links[c2])
    else:
        return sG.copy(), component_links

    # Remove swapped from possibilities
    component_links[c1].remove(ec1)
    component_links[c2].remove(ec2)

    # Perform swap
    e1 = ec1[0], ec2[1]
    e2 = ec2[0], ec1[1]

    # check if new edge not already exists
    if e1 not in sG.edges and e2 not in sG.edges:
        sG.add_edge(*e1)
        sG.add_edge(*e2)
        sG.remove_edge(*ec1)
        sG.remove_edge(*ec2)

    return sG.copy(), component_links


def write_graph(G, mypath, predefined_name="None"):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    filenumbers = [int(re.sub("[^0-9]", "", fn))
                   for fn in onlyfiles if any(i.isdigit() for i in fn)]

    if predefined_name == "None":
        if filenumbers == []:
            fpath = os.path.join(mypath, "0.gml")
        else:
            fpath = os.path.join(mypath, f"{max(filenumbers)+1}.gml")
    else:
        fpath = os.path.join(mypath, predefined_name)

    nx.write_gml(G, fpath)


def create_assortativity_networks(n_groups, n_per_group, n_samples, p_rel, runs, n_swaps, network_gen_fn="erdos"):
    """If parameterset has not been run before, makes new directory filled with
    `n_samples` directories for graphs. Then fills these directories with `runs` graphs
    named of consequtive integers (if not exist, start at zero). Note that there needs to be a
    './networks/' directory present from current file.

    Graphs are based on parameter values, and function performs a decline in assortativity
    by starting with the most constrained network, a fully assortative network (value 1).
    Then randomly picks two links in different clusters, and swaps them. If the network
    assortativity is within a certain distance from `np.linspace(0,1,n_samples)`, save
    to directory, and move on until next energy sample is close enough.

    Args:
        netgen_fn (function): Function that generates networkX network and a dictionary
            of the relationships within each energy cluster.
        n_groups (int): Amount of different clusters at beginning network.
        n_per_group (int): Number of agents per cluster.
        n_samples (int): Number of assortativity steps to sample between 0 and 1.
        p_rel (float): Chance to accept random relationship. (same as erdos-renyi)
        n_swaps (int): Number of swaps performed before assuming convergence.
        runs (int): Number of graphs generated per assortativity sample.
        plot_pearson (boolean): True if you want to plot the pearson optimization change.
        network_gen_fn (function): Network generation function that returns network and
            component link list.
    """
    # Check for directory for parameter values and create if not exist
    dirpath = f"./networks/{network_gen_fn}-{n_samples}s-{n_groups}g-{n_per_group}n-{p_rel}p"
    network_gen_fn = globals()[network_gen_fn]
    n_a_samples = np.linspace(-1, 1, n_samples)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        for e in n_a_samples:
            epath = os.path.join(dirpath, str(np.round(e, 1)))
            os.mkdir(epath)

    # For plotting pearson progressions
    ps = {}

    # Begin optimization
    for i in range(runs):
        print(f"Running graphset {i+1} out of {runs}")
        switch = 1

        # Get the sample assortativities needed
        sampled = np.round(n_a_samples, 2).tolist()
        gen_procs = {}
        aG = fully_assortative_network(
            np.linspace(0, 1, n_groups), n_per_group, p_rel, network_gen_fn)
        gen_procs["fully_assortative_network"] = aG
        gen_procs["fully_disassortative_network"] = fully_disassortative_network(
            np.linspace(0, 1, n_groups), n_per_group, p_rel, G=aG[0].copy(), component_links=aG[1])

        for netgen_fn in gen_procs:
            ps.setdefault(netgen_fn, [])
            G, component_links = gen_procs[netgen_fn]

            # Generate default network
            p = np.round(pearson(G), 2) + 0.
            sampled.remove(p)
            path = os.path.join(dirpath, str(p))
            write_graph(G, path)
            avg_degree = calc_avg_degree(G)

            pstmp = []

            for i in range(n_swaps):
                # Swap edge and assert average degree stays the same
                G, component_links = swap_edge(G, component_links)
                assert calc_avg_degree(
                    G) == avg_degree, f"Average degree changed from {calc_avg_degree(G)} to {avg_degree}, please check rewiring."

                # Calculate pearson and store graph if needed
                p = np.round(pearson(G), 2) + 0.
                pstmp.append(p)
                if p in sampled:
                    sampled.remove(p)
                    path = os.path.join(dirpath, str(p))
                    write_graph(G, path)

                if p > 0 and switch == 0:
                    break
                elif switch == 1 and p < 0:
                    switch -= 1
                    break

            # From here it's just to plot
            ps[netgen_fn].append(pstmp)
    colors = ["#45BF86", "#80A6F2"]
    fig, ax = plt.subplots(1, 1)
    for i, p in enumerate(ps):
        for l in ps[p]:
            ax.plot(l, color=colors[i])

    from matplotlib.lines import Line2D
    keys = list(gen_procs.keys())
    legend_elements = [Line2D([0], [0], color=colors[0], lw=4, label=keys[0]),
                       Line2D([0], [0], color=colors[1], lw=4, label=keys[1])]

    # Create the figure
    ax.legend(handles=legend_elements, loc="upper right")

    gf.set_frame(ax)
    gf.set_labels_and_titles(ax, f"Assortativity optimization\ntowards random",
                             "Iterations", "Assortativity", tick_size=12)
    plt.savefig(
        f"networks/imgs/{network_gen_fn}-{n_samples}s-{n_groups}g-{n_per_group}n-{p_rel}p-assort_optimization.png")
    plt.show()
    print("Finished making networks!")


def erdos(N, p, e, idx=0):
    G = nx.erdos_renyi_graph(N, p, directed=True)
    nx.set_node_attributes(
        G, {i: {"e": e, "k": e} for i in G.nodes})
    G = nx.relabel_nodes(G, {i: i+idx for i in G.nodes})
    return G, {e: list(G.edges())}


def barabasi_albert(N, m, e, idx=0):
    G = nx.barabasi_albert_graph(N, m)
    while not nx.is_connected(G):
        G = nx.barabasi_albert_graph(N, m)
    edges = list(G.edges())
    # Diffuse the tree, otherwise nodes will have scalefree incoming nodes
    # and fixed outgoing
    edges = [(edges[i][0], edges[i][1]) if i % 2 == 0 else (
        edges[i][1], edges[i][0]) for i in range(len(edges))]
    G = nx.DiGraph()
    G.add_edges_from(edges)
    G = nx.relabel_nodes(G, {i: i+idx for i in G.nodes})
    nx.set_node_attributes(
        G, {i: {"e": e, "k": e} for i in G.nodes})
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
    G = nx.relabel_nodes(G, {i: i+idx for i in G.nodes})
    nx.set_node_attributes(
        G, {i: {"e": e, "k": e} for i in G.nodes})
    return G, {e: list(G.edges())}


def focussed_assort_network_gen(aim_assort_value, e_groups, n_per_group, p_rel, n_swaps, network_gen_fn=erdos, max_rec=100, cur_rec=0, debug=False):
    if aim_assort_value == 0.:
        G, component_links = network_gen_fn(
            n_per_group * len(e_groups), p_rel, e_groups[0])
        es = np.array([[e] * n_per_group for e in e_groups]).flatten()
        np.random.shuffle(es)
        for idx, node in enumerate(G.nodes):
            G.nodes[node]["e"] = es[idx]
            G.nodes[node]["k"] = es[idx]
        return G, component_links

    G, component_links = fully_assortative_network(
        e_groups, n_per_group, p_rel, network_gen_fn)

    if aim_assort_value < 0:
        G, component_links = fully_disassortative_network(
            e_groups, n_per_group, p_rel, G=G.copy(), component_links=component_links, network_gen_fn=network_gen_fn)

    if aim_assort_value == -1. or aim_assort_value == 1.:
        return G, component_links
    # Generate default network
    p = np.round(pearson(G), 2) + 0.
    avg_degree = calc_avg_degree(G)

    if debug:
        ps = [p]

    for i in range(n_swaps):
        # Swap edge and assert average degree stays the same
        G, component_links = swap_edge(G, component_links)
        assert calc_avg_degree(
            G) == avg_degree, f"Average degree changed from {calc_avg_degree(G)} to {avg_degree}, please check rewiring."

        # Calculate pearson and store graph if needed
        p = np.round(pearson(G), 2) + 0.
        if debug:
            ps.append(p)
        if np.round(p, 3) == aim_assort_value:
            if debug:
                plt.plot(ps)
                plt.show()
            return G, component_links

    if debug:
        plt.plot(ps)
        plt.show()
    if cur_rec == max_rec:
        raise Exception("Exceeded max recursion.")
    return focussed_assort_network_gen(aim_assort_value, e_groups, n_per_group, p_rel, n_swaps, network_gen_fn=network_gen_fn, cur_rec=cur_rec+1, max_rec=max_rec)


def fully_assortative_network(e_groups, n_per_group, p, network_gen_fn=erdos):
    G, component_links = network_gen_fn(n_per_group, p, e_groups[0])
    for idx, e in enumerate(e_groups[1:]):
        tmp, e_c_list = network_gen_fn(n_per_group, p, e, (idx+1)*n_per_group)
        G = nx.compose(G, tmp)

        component_links = {**component_links, **e_c_list}
    assert pearson(G) == 1, "Network assortativity is fully assortative."
    return G, component_links


def fully_disassortative_network(e_groups, n_per_group, p, G="", component_links="", network_gen_fn=erdos):
    if G == "" or component_links == "":
        G, component_links = fully_assortative_network(
            e_groups, n_per_group, p, network_gen_fn=network_gen_fn)
    initial_ad = calc_avg_degree(G)

    pairs = [(e_g, e_groups[-(1+idx)]) for idx, e_g in enumerate(e_groups)]
    nodes = {e: [node for node in G.nodes if G.nodes[node]["e"] == e]
             for e in e_groups}

    for pair in pairs:
        options = [link[1] for link in component_links[pair[1]]]
        for link in component_links[pair[0]]:
            for i in range(100):
                if options != []:
                    to_link = random.choice(options)
                    options.remove(to_link)
                else:
                    to_link = random.choice(nodes[pair[1]])

                if (link[0], to_link) not in list(G.edges):
                    G.add_edge(link[0], to_link)
                    G.remove_edge(*link)
                    break

    component_links = {e: list(chain.from_iterable(
        [list(G.out_edges(node)) for node in nodes[e]])) for e in e_groups}

    # Perform some tests
    assert pearson(G) == -1., "Network assortativity is not -1"
    assert calc_avg_degree(G) == initial_ad, "Average degree changed"
    return G, component_links


def reshuffle_var(G_old, var):
    G = G_old.copy()
    var_values = np.array([G.nodes[node][var] for node in G.nodes])
    np.random.shuffle(var_values)
    for idx, node in enumerate(G.nodes()):
        G.nodes[node][var] = var_values[idx]
    return G
