import networkx as nx
import numpy as np


class mainModel:
    def __init__(self, G, h, beta, point, alpha=3):
        self.G = G
        self.h = h

        # Parameters
        self.alpha = alpha
        self.beta = beta
        self.pc, self.pb, self.pec = point

        # Check input values
        if np.round(sum(point), 3) != 1:
            raise ValueError(
                f"Parameters pc+pb+pec != 1 for point({point}), relative strengths cannot exceed or be lower than one."
            )

        assert isinstance(
            G, nx.classes.digraph.DiGraph
        ), "Given graph is not a NetworkX DiGraph."
        assert all(
            ["e" in G.nodes[i] for i in G.nodes]
        ), "Nodes in graph don't have energy variable."
        assert all(
            ["k" in G.nodes[i] for i in G.nodes]
        ), "Nodes in graph don't have connectivity variable."

    ##########################
    # Simulate next timestep #
    ##########################
    def next(self):
        # Initiate next snapshot
        next_timestep = {}

        # Calculate values for next timestep
        for node_i in self.G.nodes():
            # Calculate next step
            next_node_value = self.calcNodeStep(node_i)

            # Create simultaneous update from snapshot
            next_timestep.setdefault(node_i, {})
            next_timestep[node_i]["k"] = next_node_value[0]
            next_timestep[node_i]["e"] = next_node_value[1]

        for node_i in self.G.nodes():
            for variable in next_timestep[node_i]:
                self.G.nodes[node_i][variable] = next_timestep[node_i][variable]

    ####################################
    # Little shorthand for running sim #
    ####################################
    def run_for_n_steps(self, n_steps):
        for _ in range(n_steps):
            self.next()

    ################################
    # Calculate next step for node #
    ################################

    def calcNodeStep(self, i):
        k = self.G.nodes[i]["k"]
        e = self.G.nodes[i]["e"]

        # Calculate connectivity derivative
        dk = e - k * self.beta

        if self.G.in_degree(i) != 0:
            csum = []
            bsum = []
            ecsum = []

            for j in self.G.in_edges(i):
                csum.append(self.G.nodes[j[0]]["k"])
                bsum.append(
                    self.G.nodes[i]["e"]
                    * ((self.G.nodes[j[0]]["e"] - 0.5))
                    / self.G.out_degree(j[0])
                )
                ecsum.append(self.G.nodes[j[0]]["e"])

            christakis_conjecture = sum(
                [
                    self.pc * (self.G.nodes[i]["k"] - sum(csum) / self.G.in_degree(i)),
                    self.pb * (sum(bsum) / self.G.in_degree(i)),
                    self.pec
                    * (
                        (np.mean(ecsum) - self.G.nodes[i]["e"])  # / self.G.in_degree(i)
                    ),  # mayyybe change this indegree, and change the theta 0.4 to 0.5
                ]
            ) + (np.random.normal(0, 0.02) * np.sqrt(self.h))

            de = christakis_conjecture * e * (1 - e)
        else:
            de = 0

        # Return next connectivity and energy for next time step
        new_k, new_e = np.array([k, e]) + self.h * (np.array([dk, de]))

        # Add noise so they won't just converge to a artificial point
        # new_e += np.random.normal(0, 0.01)
        # if new_e > 1:
        #     new_e = 1
        # elif new_e < 0:
        #     new_e = 0

        return new_k, new_e

    ##########################
    # Define model functions #
    ##########################
    def cognitive_old(self, i):
        return self.G.nodes[i]["k"] - self.beta / self.alpha

    def cognitive(self, i):
        if self.G.in_degree(i) == 0:
            return 0

        # print(self.G.nodes[i]["k"], ((sum([self.G.nodes[j[0]]["k"] for j in self.G.in_edges(
        #     i)])/self.G.in_degree(i))*np.random.normal(1, 0.1)))

        # *np.random.normal(1, 0.05))
        return self.G.nodes[i]["k"] - (
            (
                sum([self.G.nodes[j[0]]["k"] for j in self.G.in_edges(i)])
                / self.G.in_degree(i)
            )
        )

    def behavior(self, i):
        # Check if node has neighbors
        if self.G.in_degree(i) == 0:
            return 0

        return sum(
            [
                self.G.nodes[i]["e"]
                * ((self.G.nodes[j[0]]["e"] - 0.4) / self.G.out_degree(j[0]))
                for j in self.G.in_edges(i)
            ]
        ) / self.G.in_degree(i)

    def emotional_contagion(self, i):
        # Check if node has neighbors
        if self.G.in_degree(i) == 0:
            return 0

        return (
            2
            * sum(
                [
                    ((self.G.nodes[i]["e"] + self.G.nodes[j[0]]["e"]) / 2)
                    - self.G.nodes[i]["e"]
                    for j in self.G.in_edges(i)
                ]
            )
            / self.G.in_degree(i)
        )

    # def cognitive(self, i):
    #     if self.G.out_degree(i) == 0:
    #         return 0

    #     # print(self.G.nodes[i]["k"], ((sum([self.G.nodes[j[0]]["k"] for j in self.G.in_edges(
    #     #     i)])/self.G.in_degree(i))*np.random.normal(1, 0.1)))

    #     # *np.random.normal(1, 0.05))
    #     return self.G.nodes[i]["k"] - ((sum([self.G.nodes[j[0]]["k"] for j in self.G.out_edges(i)])/self.G.out_degree(i)))

    # def behavior(self, i):
    #     # Check if node has neighbors
    #     if self.G.out_degree(i) == 0:
    #         return 0

    #     external_influence = []
    #     for j in self.G.out_edges(i):
    #         try:
    #             self.G.nodes[i]["e"] * ((self.G.nodes[j[1]]
    #                                      ["e"] - (1/3)) / self.G.in_degree(j[1]))
    #         except ZeroDivisionError:
    #             print(f"{j=}")
    #             print(f"{self.G.nodes[j[1]]=}")
    #             print(f"{self.G.in_degree(j[1])=}")
    #             print(f"{self.G.in_edges(j[1])=}")

    #             print(f"{i=}")
    #             print(f"{self.G.nodes[i]=}")
    #             print(f"{self.G.out_degree(i)=}")
    #             print(f"{self.G.out_edges(i)=}")

    #     return sum(external_influence) / self.G.out_degree(i)
    #     # return sum([
    #     #     self.G.nodes[i]["e"] * ((self.G.nodes[j[0]]["e"] - (1/3)) / self.G.in_degree(j[0])) for j in self.G.out_edges(i)
    #     # ]) / self.G.out_degree(i)

    # def emotional_contagion(self, i):
    #     # Check if node has neighbors
    #     if self.G.out_degree(i) == 0:
    #         return 0

    #     return 2 * sum([((self.G.nodes[i]["e"] + self.G.nodes[j[0]]["e"]) / 2) - self.G.nodes[i]["e"]
    #                     for j in self.G.out_edges(i)]) / self.G.out_degree(i)

    # def calcNodeStep(self, i):
    #     k = self.G.nodes[i]["k"]
    #     e = self.G.nodes[i]["e"]

    #     # Calculate connectivity derivative
    #     dk = e - k * self.beta

    #     # Calculate energy derivative
    #     christakis_conjecture = sum([
    #         self.pc * self.cognitive(i), self.pb * self.behavior(i),
    #         self.pec * self.emotional_contagion(i)
    #     ])

    #     de = christakis_conjecture * e * (1 - e)

    #     # Return next connectivity and energy for next time step
    #     new_k, new_e = np.array([k, e]) + self.h * \
    #         (np.array([dk, de]) +
    #          (np.random.normal(0, 0.01))) #* np.sqrt(self.h)))

    #     # Add noise so they won't just converge to a artificial point
    #     # new_e += np.random.normal(0, 0.01)
    #     if new_e > 1:
    #         new_e = 1
    #     elif new_e < 0:
    #         new_e = 0

    #     return new_k, new_e
