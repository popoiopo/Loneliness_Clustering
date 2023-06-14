import researchhelper.visualize.general_formatting as gf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import networkx as nx

from functions.misc import bootstrap_resample_data


def plot_timeseries(ax, data, a, data_std=[], cmap="tab10", vmin=-1, vmax=1):
    if len(data) > 0:
        cmap = plt.get_cmap(cmap)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        # ax.plot(data.T, color=cmap(norm(a)), alpha=0.2)
        data_mean = np.mean(data, axis=0)
        if data_std == []:
            data_std = np.std(data, axis=0)
        else:
            data_std = np.mean(data_std, axis=0)
        ax.plot(data_mean, color=cmap(norm(a)), zorder=11)
        ax.fill_between(
            range(len(data_mean)),
            data_mean - data_std,
            data_mean + data_std,
            color=cmap(norm(a)),
            zorder=10,
            alpha=0.5,
        )


def set_labels_and_titles(
    ax, title, xlabel, ylabel, tick_size=20, label_size=20, title_size=22
):
    """Set the labels and titles of the graph including it's (tick)sizes.

    Parameters
    ----------
    ax : matplotlib axis object
        Standard matplotlib axis object.
        E.g., as from output of `fig, ax = plt.subplots()`
    title : str
        Set graph title.
    xlabel : str
        Set label for a-axis.
    ylabel : str
        Set label for y-axis.
    tick_size : int
        Set size for the ticks. Defaults to 20.
    label_size : int
        Set size for the axes labels. Defaults to 20.
    title_size : int
        Set size for the graph title. Defaults to 22.

    """
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(tick_size)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(tick_size)

    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.set_title(title, fontsize=title_size)


def plot_metric_data(data, energies, cols, name):
    if len(data) == 0:
        rowcol = (1, 1)
    else:
        rowcol = (int(np.ceil(len(data) / cols)), cols)
    fig, ax = plt.subplots(*rowcol, figsize=(5 * rowcol[1], 4 * rowcol[0]))

    ordered_ids = np.argsort(energies)

    for idx, d in enumerate(data):
        if rowcol == (1, 1):
            axis = ax
        elif rowcol[1] == 1 or rowcol[0] == 1:
            axis = ax[idx]
        else:
            axis = ax[int(np.floor(idx / cols)), idx % cols]
        x = np.array(energies)[ordered_ids]
        y = np.mean(data[d], axis=0)
        axis.plot(x, y[ordered_ids])
        gf.set_frame(axis)
        gf.set_labels_and_titles(axis, f"{d}", "", "", tick_size=12)
    plt.tight_layout()
    plt.savefig(name)
    plt.show()


def plot_triangle_heatmap(gm, group):
    colormap = cm.rainbow
    normalize = mcolors.Normalize(vmin=-1, vmax=1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6.5))
    for c in gm.coleman:
        means, mean_of_means, se_of_means = bootstrap_resample_data(
            [d[group] for d in gm.coleman[c]], gm.conf.n_bootstrap_resample
        )
        x, y = get_xy(*[float(i) for i in c.split(",")])
        values0 = np.array([val[0] for val in gm.coleman[c]])
        values1 = np.array([val[1] for val in gm.coleman[c]])
        if np.isnan(mean_of_means):
            if np.isnan(values0).all():
                color = "lightgrey"
            elif np.isnan(values1).all():
                color = "darkgrey"
        else:
            color = colormap(normalize(mean_of_means))
        ax.scatter(x, y, color=color, norm=normalize, s=30)

    # setup the colorbar
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(np.linspace(-1, 1, 100))
    plt.colorbar(scalarmappaple, orientation="horizontal", pad=0.05)

    locs = [(1, 0), (1, 1), (0, 0), (1, 0)]
    ps = [point_on_triangle(*l) for l in locs]
    x, y, c = zip(*ps)
    ax.plot(x, y, color="grey", alpha=1)
    ax.text(x[0] - 0.02, y[0], "Behavior", ha="right")
    ax.text(x[1], y[1] + 0.04, "Cognitive", ha="center")
    ax.text(x[2] + 0.02, y[2], "Contagion")

    plt.axis("off")
    plt.show()


def point_on_triangle(x, y):
    """
    Get point on equilateral unit triangle mapped from x,y coordinates.
    """
    pt1, pt2, pt3 = (0, 0), (0.5, np.sqrt(3) / 2), (1, 0)
    q = abs(x - y)
    s, t, u = q, 0.5 * (x + y - q), 1 - 0.5 * (q + x + y)
    return (
        s * pt1[0] + t * pt2[0] + u * pt3[0],
        s * pt1[1] + t * pt2[1] + u * pt3[1],
        [s, t, u],
    )


def get_xy(s, t, u):
    pt1, pt2, pt3 = (0, 0), (0.5, np.sqrt(3) / 2), (1, 0)
    return (
        s * pt1[0] + t * pt2[0] + u * pt3[0],
        s * pt1[1] + t * pt2[1] + u * pt3[1],
    )


def nx_network_mp4(Gs, labels, save_path, remove_inactive_nodes=False):
    """Create an MP4 from the network dynamics over time.

    Parameters
    ----------
    full_graph : nx.Graph
        Networkx graph with all nodes and edges.
    Gs : List[nx.Graph
        List of networkx graphs that change over time.
    labels : List[str
        List of labels showing the current time.
    layout : nx.__layout
        Networkx layout.
        (e.g., nx.kamada_kawai_layout, nx.spring_layout)
    save_path : str
        Where do you want to save your mp4?
    remove_inactive_nodes : bool
        True to remove inactive nodes.
        Otherwise full nodelist is always visible. Defaults to False.

    Returns
    -------
    None

    """
    assert len(Gs) == len(labels), "Desired (len(Gs) == len(labels))"

    cmap = plt.get_cmap("viridis")
    pos = nx.nx_agraph.graphviz_layout(Gs[0], prog="neato")
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.axis("off")

    # Add Edges
    segments = np.array([[pos[x], pos[y]] for x, y in Gs[0].edges()])
    edges = LineCollection(segments, alpha=0.4, linewidths=2, color="#727272")
    ax.add_artist(edges)

    # Add Nodes
    sizes = [100 + (Gs[0].nodes[node]["k"] * 300) for node in Gs[0].nodes]
    colors = [cmap(Gs[0].nodes[node]["e"]) for node in Gs[0].nodes]
    coordinates = np.array([i for i in pos.values()]).T
    nodes = ax.scatter(*coordinates, s=sizes, alpha=1, color=colors)

    # Add Time indication
    text = ax.text(1.1, 1, labels[0], ha="right", fontsize=56, color="C1", wrap=True)

    # Combine elements into list to pass later on
    actors = [edges, nodes, text]

    def update(i: int, edges, nodes, text, pos):
        """

        Parameters
        ----------
        i: int
            Index of frames.
        edges : np.array
            List of edge locations of previous step.
        nodes : np.array
            List of node locations of previous step.
        text : str
            Text to be shown in upper right corner at each update.
        pos : dict
            Networkx - Matplotlib positional layout dictionary.
        ax : matplotlib.axis
            Matplotlib axis to plot in.


        Returns
        -------
        None

        """

        segments = [[pos[x], pos[y]] for x, y in Gs[i].edges()]
        edges.set_paths(segments)

        sizes = [100 + (Gs[i].nodes[node]["k"] * 300) for node in Gs[i].nodes]
        colors = [cmap(Gs[i].nodes[node]["e"]) for node in Gs[i].nodes]
        nodes.set_color(colors)
        nodes.set_sizes(sizes)
        text.set_text(labels[i])

    # Create animation and save it
    ani = animation.FuncAnimation(
        fig, update, frames=len(Gs), fargs=(*actors, pos), interval=350, repeat=False
    )
    ani.save(save_path)
    return ani


def plot_grid(ax, G, with_labels=False, title="", cmap_name="viridis"):
    """Plot grid network.

    Parameters
    ----------
    ax : matplotlib.axis
        Matplotlib axis to plot in.
    G : nx.DiGraph
        Directed networkx graph
    with_labels : boolean
        True or false boolean on whether or not to show labels.
        (Default value = False)
    title : str
        Title of the plot. (Default value = '')

    Returns
    -------
    None

    """
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")

    sizes = [50 for node in G.nodes]
    colors = [cmap(norm(G.nodes[node]["e"])) for node in G.nodes]
    nx.draw(
        G,
        pos,
        node_size=sizes,
        edge_color="gray",
        node_color=colors,
        with_labels=with_labels,
        alpha=0.9,
        font_size=14,
        ax=ax,
    )
    ax.set_title(title)
    return


def get_xy(s, t, u):
    pt1, pt2, pt3 = (0, 0), (0.5, np.sqrt(3) / 2), (1, 0)
    return (
        s * pt1[0] + t * pt2[0] + u * pt3[0],
        s * pt1[1] + t * pt2[1] + u * pt3[1],
    )


def plot_triangle(
    data,
    e_var,
    assort,
    network_gen_fn,
    n_samples,
    n_per_group,
    p_rel,
    n_runs,
    path_name_add="",
    save_fig=False,
):
    cmap = cm.rainbow
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6.5))

    def size(x):
        return 15 + (x * 240)

    for p in data:
        ax.scatter(
            p[0],
            p[1],
            color=cmap(norm(np.nanmean(data[p]))),
            s=size(np.nanmean(e_var[p])),
        )

    # setup the colorbar
    scalarmappaple = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalarmappaple.set_array(np.linspace(-1, 1, 100))
    plt.colorbar(scalarmappaple, orientation="horizontal", pad=0.05)

    locs = [(1, 0), (1, 1), (0, 0), (1, 0)]
    ps = [point_on_triangle(*l) for l in locs]
    x, y, c = zip(*ps)
    ax.plot(x, y, color="grey", alpha=1)
    ax.text(x[0] - 0.02, y[0], "Behavior", ha="right")
    ax.text(x[1], y[1] + 0.04, "Cognitive", ha="center")
    ax.text(x[2] + 0.02, y[2], "Contagion")
    ax.text(
        -0.15,
        y[1],
        "Assortativity\n" + r"$t_0$=" + f"{assort}",
        ha="left",
        fontdict={"fontsize": 14},
    )

    plt.axis("off")
    if save_fig:
        save_path = f"networks/imgs/{network_gen_fn.__name__}-{assort}a-{n_samples}s-{n_per_group}n-{p_rel}p_{n_runs}nr-{len(data)}-{path_name_add}npoints_triangle.png"
        plt.savefig(save_path)
    else:
        plt.show()
