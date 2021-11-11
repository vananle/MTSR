import itertools

import numpy as np
import pulp as pl

from . import util


class OneStepSRSolver:

    def __init__(self, G, segments):
        """
        G: networkx Digraph, a network topology
        """
        self.G = G
        self.num_node = G.number_of_nodes()
        self.segments = segments
        self.problem = None
        self.var_dict = None
        self.solution = None
        self.status = None

    def create_problem(self, tm):
        # 1) create optimization model
        problem = pl.LpProblem('SegmentRouting', pl.LpMinimize)
        theta = pl.LpVariable(name='theta', lowBound=0.0, cat='Continuous')

        x = pl.LpVariable.dicts(name='x',
                                indexs=np.arange(self.num_node ** 3),
                                cat='Binary')

        # 2) objective function
        # minimize maximum link utilization
        problem += theta

        # 3) constraint function
        for u, v in self.G.edges:
            capacity = self.G.get_edge_data(u, v)['capacity']
            load = pl.lpSum(
                x[util.flatten_index(i, j, k, self.num_node)] * tm[i, j] * util.g(self.segments, i, j, k, u, v)
                for i, j, k in
                itertools.product(range(self.num_node), range(self.num_node), range(self.num_node)))
            problem += load <= theta * capacity

        # 3) constraint function
        # ensure all traffic are routed
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):
            problem += pl.lpSum(x[util.flatten_index(i, j, k, self.num_node)] for k in range(self.num_node)) == 1

        return problem, x

    def extract_solution(self, problem):
        # extract solution
        self.var_dict = {}
        for v in problem.variables():
            self.var_dict[v.name] = v.varValue

        self.solution = np.empty([self.num_node, self.num_node, self.num_node])
        for i, j, k in itertools.product(range(self.num_node), range(self.num_node), range(self.num_node)):
            index = util.flatten_index(i, j, k, self.num_node)
            self.solution[i, j, k] = self.var_dict['x_{}'.format(index)]

    def evaluate(self, tm, solution):
        # extract utilization
        mlu = 0
        for u, v in self.G.edges:
            load = sum([solution[i, j, k] * tm[i, j] * util.g(self.segments, i, j, k, u, v) for i, j, k in
                        itertools.product(range(self.num_node), range(self.num_node), range(self.num_node))])
            capacity = self.G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            self.G[u][v]['utilization'] = utilization
            if utilization >= mlu:
                mlu = utilization
        return mlu

    def extract_status(self, problem):
        self.status = pl.LpStatus[problem.status]

    def init_solution(self):
        solution = np.zeros([self.num_node, self.num_node, self.num_node])
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):
            solution[i, j, i] = 1

        return solution

    def solve(self, tm):
        problem, x = self.create_problem(tm)
        self.solution = self.init_solution()
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
