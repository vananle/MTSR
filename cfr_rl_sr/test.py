from __future__ import print_function

import itertools
import os.path

import networkx as nx
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from joblib import delayed, Parallel
from tqdm import trange

from config import get_config
from env import Environment
from game_mssr import CFRRL_Game
from model import Network

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')

config_gb = get_config(FLAGS) or FLAGS


def count_rc_optimal(solutions, flows_list, lp_links, method):
    rc = []

    for i in range(len(solutions) - 1):
        count = 0
        solution_0 = solutions[i][method]
        solution_1 = solutions[i + 1][method]

        for flow_idx in flows_list:
            for e in lp_links:
                if solution_1[flow_idx, e[0], e[1]] - solution_0[flow_idx, e[0], e[1]] != 0.0:
                    count += 1
                    break

        rc.append(count)
    rc = np.asarray(rc)
    return rc


def count_rc_cfr(solutions, crit_pairs, lp_links, method):
    rc = []
    for i in range(len(solutions) - 1):
        count = 0
        solution_0 = solutions[i][method]
        solution_1 = solutions[i + 1][method]
        pairs_0 = crit_pairs[i][method]
        pairs_1 = crit_pairs[i + 1][method]

        intersec_flows = np.intersect1d(pairs_0, pairs_1)

        # counting the number of flows added to the list
        new_cflows = np.setdiff1d(pairs_1, intersec_flows)
        count += new_cflows.shape[0]

        # counting the number of flows that have path changed
        for flow_idx in intersec_flows:
            for e in lp_links:
                if solution_1[flow_idx, e[0], e[1]] - solution_0[flow_idx, e[0], e[1]] != 0.0:
                    count += 1
                    break

        rc.append(count)
    rc = np.asarray(rc)

    return rc


def count_rc(solutions, crit_pairs, lp_links, num_pairs):
    method = ['cfr-rl', 'cfr-topk', 'topk', 'optimal']
    num_rc = {}
    for m in range(len(method)):

        if method[m] == 'optimal':
            flows_list = np.arange(num_pairs)
            rc = count_rc_optimal(solutions=solutions, flows_list=flows_list,
                                  lp_links=lp_links, method=m)
            num_rc[method[m]] = rc

        else:
            rc = count_rc_cfr(solutions=solutions, crit_pairs=crit_pairs,
                              lp_links=lp_links, method=m)
            num_rc[method[m]] = rc

    return num_rc


def shortest_path(graph, source, target):
    return nx.shortest_path(graph, source=source, target=target, weight='weight')


def get_paths_from_solution(graph, solution, i, j):
    n = solution.shape[0]
    if i == j:
        list_k = [i]
    elif len(solution[i, j]) < n:  # handling srls solution
        if len(solution[i, j]) == 2:
            list_k = [i]
        else:
            list_k = solution[i, j][1:-1]
    else:  # handling solution in shape of nxnxn
        list_k = np.where(solution[i, j] == 1.0)[0]

    paths = []
    for k in list_k:
        path = []
        path += shortest_path(graph, i, k)[:-1]
        path += shortest_path(graph, k, j)
        paths.append(path)
    return paths


def get_route_changes(routings, graph):
    route_changes = np.zeros(shape=(routings.shape[0] - 1))
    for t in range(routings.shape[0] - 1):
        _route_changes = 0
        for i, j in itertools.product(range(routings.shape[1]), range(routings.shape[2])):
            path_t_1 = get_paths_from_solution(graph, routings[t + 1], i, j)
            path_t = get_paths_from_solution(graph, routings[t], i, j)
            if path_t_1 != path_t:
                _route_changes += 1

        route_changes[t] = _route_changes
    return route_changes


def sim_cfr_rl(network, game):
    mlus = []
    solutions = []
    p_solution = game.mssr_cfr_solver.initialize()
    for tm_idx in game.tm_indexes:
        state = game.get_state(tm_idx)
        policy = network.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        actions = policy.argsort()[-game.max_moves:]
        u, solution, use_pSolution = game.evaluate_cfr_rl(tm_idx, pSolution=p_solution, actions=actions)
        mlus.append(u)
        solutions.append(np.copy(solution))
        p_solution = solution
        description = 'CFR_RL| ' + str(tm_idx) + ', ' + str(u) + '   ---  use pSolution: ' + str(use_pSolution)
        print(description)

    solutions = np.asarray(solutions)
    mlus = np.asarray(mlus)
    num_rc = get_route_changes(solutions, game.DG)
    if not os.path.exists('./results/'):
        os.makedirs('./results')
    np.save('./results/{}_{}_{}_mlu'.format(config_gb.dataset, config_gb.max_moves, "cfr_rl"), mlus)
    np.save('./results/{}_{}_{}_rc'.format(config_gb.dataset, config_gb.max_moves, "cfr_rl"), num_rc)

    print('----------------------------------- OVERALL RESULTS ---------------------------------------------------')
    print('MLU CFR_RL: ', np.mean(mlus))
    print('RC CFR_RL: Total: {}  -  Avg: {}'.format(np.sum(num_rc), np.mean(num_rc)))


def sim_crit_topk(config, game):
    def parallel_sim_crit_topk(start_idx, stop_idx):
        stop_idx = stop_idx if stop_idx < len(game.tm_indexes) else len(game.tm_indexes)
        print(start_idx)
        print(stop_idx)
        mlus = []
        solutions = []
        p_solution = game.mssr_cfr_solver.initialize()
        for tm_idx in range(start_idx, stop_idx, 1):
            u, solution, use_pSolution = game.evaluate_crit_topk(tm_idx, pSolution=p_solution)
            mlus.append(u)
            solutions.append(np.copy(solution))
            description = 'CRIT-TOPK| ' + str(tm_idx) + ', ' + str(u) + '   ---  use pSolution: ' + str(use_pSolution)
            print(description)

        solutions = np.asarray(solutions)
        mlus = np.asarray(mlus)
        num_rc = get_route_changes(solutions, game.DG)
        if not os.path.exists('./results/'):
            os.makedirs('./results')
        np.save('./results/{}_{}_{}_mlu_{}'.format(config.dataset, config.max_moves, "cfr_topk", start_idx), mlus)
        np.save('./results/{}_{}_{}_rc_{}'.format(config.dataset, config.max_moves, "cfr_topk", start_idx), num_rc)

    p = int(len(game.tm_indexes) / os.cpu_count())
    Parallel(n_jobs=os.cpu_count())(delayed(parallel_sim_crit_topk)(start_idx=start_idx, stop_idx=start_idx + p)
                                    for start_idx in range(0, len(game.tm_indexes), p))

    mlus, rcs = [], []
    for start_idx in range(0, len(game.tm_indexes), p):
        mlu = np.load('./results/{}_{}_{}_mlu_{}.npy'.format(config.dataset, config.max_moves, "cfr_topk", start_idx))
        rc = np.load('./results/{}_{}_{}_rc_{}.npy'.format(config.dataset, config.max_moves, "cfr_topk", start_idx))
        mlus.append(mlu)
        rcs.append(rc)

    mlus = np.asarray(mlus)
    rcs = np.asarray(rcs)
    np.save('./results/{}_{}_{}_mlu'.format(config.dataset, config.max_moves, "cfr_topk"), mlus)
    np.save('./results/{}_{}_{}_rc'.format(config.dataset, config.max_moves, "cfr_topk"), rcs)

    print('----------------------------------- OVERALL RESULTS ---------------------------------------------------')
    print('MLU CFR-TOPK: ', np.mean(mlus))
    print('RC CFR-TOPK: Total: {}  -  Avg: {}'.format(np.sum(rcs), np.mean(rcs)))


def sim_topk(config, game):
    def parallel_sim_topk(start_idx, stop_idx):
        stop_idx = stop_idx if stop_idx < len(game.tm_indexes) else len(game.tm_indexes)
        print(start_idx)
        print(stop_idx)

        mlus = []
        solutions = []
        p_solution = game.mssr_cfr_solver.initialize()
        for tm_idx in range(start_idx, stop_idx, 1):
            u, solution, use_pSolution = game.evaluate_topk(tm_idx, pSolution=p_solution)
            mlus.append(u)
            solutions.append(np.copy(solution))
            p_solution = solution
            description = 'TOPK| ' + str(tm_idx) + ', ' + str(u) + '   ---  use pSolution: ' + str(use_pSolution)
            print(description)

        solutions = np.asarray(solutions)
        mlus = np.asarray(mlus)
        num_rc = get_route_changes(solutions, game.DG)

        if not os.path.exists('./results/'):
            os.makedirs('./results')
        np.save('./results/{}_{}_{}_mlu_{}'.format(config.dataset, config.max_moves, "topk", start_idx), mlus)
        np.save('./results/{}_{}_{}_rc_{}'.format(config.dataset, config.max_moves, "topk", start_idx), num_rc)

    p = int(len(game.tm_indexes) / os.cpu_count())
    Parallel(n_jobs=os.cpu_count())(delayed(parallel_sim_topk)(start_idx=start_idx, stop_idx=start_idx + p)
                                    for start_idx in range(0, len(game.tm_indexes), p))

    mlus, rcs = [], []
    for start_idx in range(0, len(game.tm_indexes), p):
        mlu = np.load('./results/{}_{}_{}_mlu_{}.npy'.format(config.dataset, config.max_moves, "topk", start_idx))
        rc = np.load('./results/{}_{}_{}_rc_{}.npy'.format(config.dataset, config.max_moves, "topk", start_idx))
        mlus.append(mlu)
        rcs.append(rc)

    mlus = np.asarray(mlus)
    rcs = np.asarray(rcs)
    np.save('./results/{}_{}_{}_mlu'.format(config.dataset, config.max_moves, "topk"), mlus)
    np.save('./results/{}_{}_{}_rc'.format(config.dataset, config.max_moves, "topk"), rcs)

    print('----------------------------------- OVERALL RESULTS ---------------------------------------------------')
    print('MLU TOPK: ', np.mean(mlus))
    print('RC TOPK: Total: {}  -  Avg: {}'.format(np.sum(rcs), np.mean(rcs)))


def sim_optimal(config, game):
    def parallel_sim_optimal(start_idx, stop_idx):
        stop_idx = stop_idx if stop_idx < len(game.tm_indexes) else len(game.tm_indexes)
        print(start_idx)
        print(stop_idx)

        mlus = []
        solutions = []
        p_solution = game.mssr_cfr_solver.initialize()
        for tm_idx in range(start_idx, stop_idx, 1):
            u, solution = game.evaluate_optimal(tm_idx, pSolution=p_solution)
            mlus.append(u)
            solutions.append(np.copy(solution))
            p_solution = solution
            description = 'OPTIMAL| ' + str(tm_idx) + ', ' + str(u)
            print(description)
            # iterator.set_description(description)

        solutions = np.asarray(solutions)
        mlus = np.asarray(mlus)
        num_rc = get_route_changes(solutions, game.DG)
        print('----------------------------------- OVERALL RESULTS ---------------------------------------------------')
        print('MLU OPTIMAL: ', np.mean(mlus, axis=0))
        print('RC OPTIMAL: Total: {}  -  Avg: {}'.format(np.sum(num_rc), np.mean(num_rc)))

        if not os.path.exists('./results/'):
            os.makedirs('./results')
        np.save('./results/{}_{}_{}_mlu_{}'.format(config.dataset, config.max_moves, "optimal", start_idx), mlus)
        np.save('./results/{}_{}_{}_rc_{}'.format(config.dataset, config.max_moves, "optimal", start_idx), num_rc)

    p = int(len(game.tm_indexes) / os.cpu_count())
    Parallel(n_jobs=os.cpu_count())(delayed(parallel_sim_optimal)(start_idx=start_idx, stop_idx=start_idx + p)
                                    for start_idx in range(0, len(game.tm_indexes), p))

    mlus, rcs = [], []
    for start_idx in range(0, len(game.tm_indexes), p):
        mlu = np.load('./results/{}_{}_{}_mlu_{}.npy'.format(config.dataset, config.max_moves, "optimal", start_idx))
        rc = np.load('./results/{}_{}_{}_rc_{}.npy'.format(config.dataset, config.max_moves, "optimal", start_idx))
        mlus.append(mlu)
        rcs.append(rc)

    mlus = np.asarray(mlus)
    rcs = np.asarray(rcs)
    np.save('./results/{}_{}_{}_mlu'.format(config.dataset, config.max_moves, "optimal"), mlus)
    np.save('./results/{}_{}_{}_rc'.format(config.dataset, config.max_moves, "optimal"), rcs)
    print('----------------------------------- OVERALL RESULTS ---------------------------------------------------')
    print('MLU OPTIMAL: ', np.mean(mlus))
    print('RC OPTIMAL: Total: {}  -  Avg: {}'.format(np.sum(rcs), np.mean(rcs)))


def sim_sp(config, game):
    mlus = []
    iterator = trange(len(game.tm_indexes))
    for tm_idx in iterator:
        u = game.evaluate_sp(tm_idx)
        mlus.append(u)
        description = 'SP| ' + str(tm_idx) + ', ' + str(u)
        iterator.set_description(description)

    mlus = np.asarray(mlus)
    print('----------------------------------- OVERALL RESULTS -------------------------------------------------------')
    print('MLU SP: ', np.mean(mlus))

    if not os.path.exists('./results/'):
        os.makedirs('./results')
    np.save('./results/{}_{}_{}_mlu'.format(config.dataset, config.max_moves, "sp"), mlus)


def main(_):
    # Using cpu for testing
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.get_logger().setLevel('INFO')

    config = get_config(FLAGS) or FLAGS

    if config.eval_method == config.eval_methods[0]:
        env = Environment(config_gb, is_training=False)
        game = CFRRL_Game(config_gb, env)
        network = Network(config_gb, game.state_dims, game.action_dim, game.max_moves)

        step = network.restore_ckpt(FLAGS.ckpt)
        learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
        print('\nstep %d, learning rate: %f\n' % (step, learning_rate))
        sim_cfr_rl(network, game)

    elif config.eval_method == config.eval_methods[1]:
        env = Environment(config, is_training=False)
        game = CFRRL_Game(config, env)
        sim_crit_topk(config, game)
    elif config.eval_method == config.eval_methods[2]:
        env = Environment(config, is_training=False)
        game = CFRRL_Game(config, env)
        sim_topk(config, game)
    elif config.eval_method == config.eval_methods[3]:
        env = Environment(config, is_training=False)
        game = CFRRL_Game(config, env)
        sim_optimal(config, game)
    elif config.eval_method == config.eval_methods[4]:
        env = Environment(config, is_training=False)
        game = CFRRL_Game(config, env)
        sim_sp(config, game)
    else:
        raise NotImplementedError("Evaluation method not implemented")


if __name__ == '__main__':
    app.run(main)
