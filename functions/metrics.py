import numpy as np
import networkx as nx
from collections.abc import Iterable
from collections import Counter
import pandas as pd


def make_group_assignment(G):
    # Check who is in which group
    group_assignment = np.array([round(G.nodes[v]["e"]) for v in G.nodes()])
    theoretical_groups = np.array([0, 1])
    return group_assignment, theoretical_groups


def coleman_homophily_index(G, make_group_assignment):
    """The segregation index S^g_{coleman} for group is established to represent the propensity of an individual to create a tie to someone from the same group (i.e., the extent of homophily), as opposed to choosing randomly.

    It provides an index that varies between -1 (perfectly avoiding one's own group) and 1 (perfect segregation). The index assumes the value 0 if and only if the expected number of withingroup ties under random choice is exactly equal to the observed number of within-group ties, given the total degree of a group.

    Args:
        G (nx.DiGraph): A networkx graph.
        t (array): An array for each agent depicting which group they belong to, list of floats or ints

    Returns:
        dict: Coleman segregation index for each group
    """
    # Assign agents to groups
    t, theoretic_groups = make_group_assignment(G)

    # Filter people based on their group
    nGs = {i: np.where(t == i)[0] for i in range(0, len(set(t)))}

    # Get total size of system
    N = len(G.nodes())

    # Calculate how many within group ties we expect based on their ratio in the network
    expected_within_group_ties = {nG: sum(
        [G.out_degree(i)*((len(nGs[nG])-1)/(N-1)) for i in nGs[nG]]) for nG in nGs}

    # Calculate how many within group ties we see
    within_group_ties = {nG: sum(
        [sum([1 for n in G.neighbors(v) if t[v] == t[n]]) for v in nGs[nG]]) for nG in nGs}

    coleman = {}

    # Apply Coleman homophily index to all groups
    for nG in theoretic_groups:
        try:
            if within_group_ties[nG] < expected_within_group_ties[nG]:
                coleman[nG] = (within_group_ties[nG] -
                               expected_within_group_ties[nG]) / expected_within_group_ties[nG]
            else:
                out_degree_agents = sum(
                    [G.out_degree(i) for i in nGs[nG]]) - expected_within_group_ties[nG]
                coleman[nG] = (within_group_ties[nG] -
                               expected_within_group_ties[nG]) / out_degree_agents
        except ZeroDivisionError as e:
            coleman[nG] = np.nan
        except KeyError as k:
            coleman[nG] = np.nan
    return coleman


def pearson(G, precision=5):
    energy_links = np.array(
        [[G.nodes[nodes[0]]["e"], G.nodes[nodes[1]]["e"]] for nodes in G.edges])

    if np.all(energy_links[:, 0] == energy_links[:, 0][0]) and np.all(energy_links[:, 1] == energy_links[:, 1][0]):
        corrcoef = [[1, 1], [1, 1]]
    else:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                corrcoef = np.corrcoef(energy_links[:, 0], energy_links[:, 1])
            except Warning as e:
                print('error found:', e)
                print(energy_links)
                corrcoef = [[np.nan, np.nan], [np.nan, np.nan]]
                raise Exception("Energy links were off, check this!")

    return list(map(lambda x: list(np.around(x, precision,)), corrcoef))[0][1]


def calc_avg_degree(G):
    return sum([G.degree[i] for i in G.nodes])/len(G.nodes)


def calc_deg_assort(G):
    return nx.degree_assortativity_coefficient(G)


def calc_betweenness(G):
    return np.mean([n for n in nx.betweenness_centrality(G).values()])


def calc_mean_energy(G):
    return np.mean([G.nodes[i]["e"] for i in G.nodes])


def calc_pearson(G):
    return pearson(G)


def flatten(any_list):
    for element in any_list:
        if hasattr(element, "__iter__") and not isinstance(element, (str, bytes)):
            yield from flatten(element)
        else:
            yield element


def check_lonely(G, threshold=0.4):
    """Return dict of node energies rounded by threshold value in a graph.

    Args:
        G (nx.DiGraph): Graph of network
        threshold (float, optional): Threshold value where energie becomes rounded to 0 (e<=threshold -> 0 else, 1). Defaults to 0.5.

    Returns:
        dict: Per node entry of their rounded energy value (lonely (0) or not (1))
    """
    def is_lonely(n): return np.floor(n) if n <= threshold else np.ceil(n)
    return {i: is_lonely(G.nodes[i]["e"]) for i in G.nodes}


def dos_neighbors(G, depth=4):
    """Provide a structure for each node and others at degree of separation depth.

    Args:
        G (nx.DiGraph): Graph of network
        depth (int, optional): The degrees of separation you want lists of. Defaults to 4.

    Returns:
        dict: Dictionary providing lists of nodes from data[node] at depth d. data[node][d] = list nodes at desired degree of separation.
    """
    def get_neighbours(l): return list(
        flatten([list(nx.neighbors(G, n)) for n in l]))
    dos_n = {}
    for node in list(G.nodes):
        dos_n.setdefault(node, {})
        dos_n[node][1] = list(nx.neighbors(G, node))
        already_iterated = dos_n[node][1].copy()
        for d in range(2, depth+1):
            nd = get_neighbours(dos_n[node][d-1])
            dos_n[node][d] = list(set(nd) - set(already_iterated))
            already_iterated += nd
    return dos_n


def dos_df(dos_n, energies, states=[0, 1]):
    df = {"ego": [], "dos": [], "ego_e": [], "n_neighbors": []}
    for node in dos_n:
        for degree in dos_n[node]:
            df["ego"].append(node)
            df["dos"].append(degree)
            df["ego_e"].append(energies[node])
            energies_neighbors = [energies[n] for n in dos_n[node][degree]]
            df["n_neighbors"].append(len(energies_neighbors))
            for k in states:
                count = Counter(energies_neighbors)
                df.setdefault(f"alter_e{k}", [])
                df[f"alter_e{k}"].append(count.get(k, 0))
    counts = Counter(energies.values())
    min_energy = min(list(counts.keys()))
    expected_fraction_lonely = counts[min_energy] / len(energies)
    df = pd.DataFrame(df)
    df["alter_fraction_e0"] = df["alter_e0"]/df["n_neighbors"]
    df["difference_random"] = (
        df["alter_fraction_e0"] / expected_fraction_lonely) - 1
    df["dos_metric"] = df["difference_random"].div(
        df.groupby(["ego"])["difference_random"].transform("first"))
    return df
