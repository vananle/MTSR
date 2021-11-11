import itertools

import numpy as np
import pulp as pl

from . import util


def flatten_index(i, j, num_edge):
    return i * num_edge + j


class ObliviousRoutingSolver:

    def __init__(self, G, segments):
        """
        G: networkx Digraph, a network topology
        """
        self.G = G
        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
        # self.segments = util.get_segments(G)
        self.segments = segments
        self.num_tms = 0
        self.problem = None
        self.var_dict = None
        self.solution = None
        self.status = None

    def create_problem(self):

        # 0) initialize lookup dictionary from index i to edge u, v
        edges_dictionary = {}
        for i, (u, v) in enumerate(self.G.edges):
            edges_dictionary[i] = (u, v)

        # 1) create optimization model of dual problem
        problem = pl.LpProblem('SegmentRouting', pl.LpMinimize)
        theta = pl.LpVariable(name='theta', lowBound=0.0, cat='Continuous')
        x = pl.LpVariable.dicts(name='x', indexs=np.arange(self.num_nodes ** 3), cat='Binary')
        pi = pl.LpVariable.dicts(name='pi', indexs=np.arange(self.num_edges ** 2), lowBound=0.0)

        # 2) objective function
        # minimize maximum link utilization
        problem += theta

        # 3) constraint function 2
        for i, j in itertools.product(range(self.num_nodes), range(self.num_nodes)):  # forall ij

            for e_prime in edges_dictionary:  # forall e' = [u, v]
                u, v = edges_dictionary[e_prime]
                # sum(g_ijk(e'))*alpha_ijk
                lb = pl.lpSum(
                    [util.g(self.segments, i, j, k, u, v) * x[util.flatten_index(i, j, k, self.num_nodes)] for k in
                     range(self.num_nodes)])
                for m in range(self.num_nodes):  # forall m
                    # sum(g_ijm(e) * pi(e,e)') >= sum(g_ijk(e')) * alpha_ijk
                    problem += pl.lpSum([util.g(self.segments, i, j, m, edges_dictionary[e][0], edges_dictionary[e][1])
                                         * pi[flatten_index(e, e_prime, self.num_edges)] for e in
                                         edges_dictionary]) >= lb

        # 4) constraint function 3
        for e_prime in edges_dictionary:  # for edge e'   sum(c(e) * pi(e, e')) <= theta * c(e')
            u, v = edges_dictionary[e_prime]
            capacity_e_prime = self.G.get_edge_data(u, v)['capacity']
            problem += pl.lpSum([self.G.get_edge_data(edges_dictionary[e][0], edges_dictionary[e][1])['capacity'] *
                                 pi[flatten_index(e, e_prime, self.num_edges)] for e in edges_dictionary]) \
                       <= theta * capacity_e_prime

        # 3) constraint function 4
        for i, j in itertools.product(range(self.num_nodes),
                                      range(self.num_nodes)):  # forall ij:   sunm(alpha_ijk) == 1.0
            problem += pl.lpSum(x[util.flatten_index(i, j, k, self.num_nodes)] for k in range(self.num_nodes)) == 1

        return problem, x

    def extract_solution(self, problem):
        # extract solution
        self.var_dict = {}
        for v in problem.variables():
            self.var_dict[v.name] = v.varValue

        self.solution = np.empty([self.num_nodes, self.num_nodes, self.num_nodes])
        for i, j, k in itertools.product(range(self.num_nodes), range(self.num_nodes), range(self.num_nodes)):
            index = util.flatten_index(i, j, k, self.num_nodes)
            self.solution[i, j, k] = self.var_dict['x_{}'.format(index)]

    def evaluate(self, tm):
        # extract utilization
        mlu = 0
        for u, v in self.G.edges:
            load = sum([self.solution[i, j, k] * tm[i, j] * util.g(self.segments, i, j, k, u, v) for i, j, k in
                        itertools.product(range(self.num_nodes), range(self.num_nodes), range(self.num_nodes))])
            capacity = self.G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            self.G[u][v]['utilization'] = utilization
            if utilization >= mlu:
                mlu = utilization
        return mlu

    def extract_status(self, problem):
        self.status = pl.LpStatus[problem.status]

    def init_solution(self):
        self.solution = np.zeros([self.num_nodes, self.num_nodes, self.num_nodes])
        for i, j in itertools.product(range(self.num_nodes), range(self.num_nodes)):
            self.solution[i, j, i] = 1

    def solve(self):
        problem, x = self.create_problem()
        self.init_solution()
        problem.solve()
        self.problem = problem
        self.extract_status(problem)
        self.extract_solution(problem)

    def get_paths(self, i, j):
        if i == j:
            list_k = [i]
        else:
            list_k = np.where(self.solution[i, j] > 0)[0]
        paths = []
        for k in list_k:
            path = []
            path += util.shortest_path(self.G, i, k)[:-1]
            path += util.shortest_path(self.G, k, j)
            paths.append((k, path))
        return paths
