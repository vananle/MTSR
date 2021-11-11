from tqdm import tqdm

from .ls2sr import LS2SRSolver
from .max_step_sr import MaxStepSRSolver
from .multi_step_sr import MultiStepSRSolver
from .oblivious_routing import ObliviousRoutingSolver
from .one_step_sr import OneStepSRSolver
from .srls import SRLS, CapacityData, ShortestPaths
from .util import *


def calculate_lamda(y_gt):
    sum_max = np.sum(np.max(y_gt, axis=1))
    maxmax = np.max(y_gt)
    return sum_max / maxmax


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


def get_route_changes_heuristic(routings):
    route_changes = []
    for t in range(routings.shape[0] - 1):
        route_changes.append(count_routing_change(routings[t + 1], routings[t]))

    route_changes = np.asarray(route_changes)
    return route_changes


def extract_results(results):
    mlus, solutions = [], []
    for _mlu, _solution in results:
        mlus.append(_mlu)
        solutions.append(_solution)

    mlus = np.stack(mlus, axis=0)
    solutions = np.stack(solutions, axis=0)

    return mlus, solutions


def save_results(log_dir, fname, mlus, route_change):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    np.save(os.path.join(log_dir, fname + '_mlus'), mlus)
    np.save(os.path.join(log_dir, fname + '_route_change'), route_change)


def prepare_te_data(x_gt, y_gt, yhat, args):
    te_step = args.test_size if args.te_step == 0 else args.te_step
    x_gt = x_gt[0:te_step:args.seq_len_y]
    y_gt = y_gt[0:te_step:args.seq_len_y]
    if args.run_te == 'ls2sr' or args.run_te == 'onestep':
        yhat = yhat[0:te_step:args.seq_len_y]

    return x_gt, y_gt, yhat


def oblivious_routing_solver(y_gt, G, segments, te_step, args):
    solver = ObliviousRoutingSolver(G, segments)
    solver.solve()
    print('Solving Obilious Routing: Done')
    results = []

    def f(tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        tms[tms <= 0.0] = 0.0
        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        return oblivious_sr(solver, tms)

    for i in tqdm(range(te_step)):
        results.append(f(tms=y_gt[i]))

    mlu, solutions = extract_results(results)
    rc = get_route_changes(solutions, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))

    print('Oblivious              | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                              np.mean(mlu),
                                                                              np.max(mlu),
                                                                              np.std(mlu)))
    congested = mlu[mlu >= 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'or', mlu, rc)


def gwn_ls2sr(yhat, y_gt, graph, te_step, args):
    print('ls2sr_gwn_p2')
    for run_test in range(args.nrun):

        results = []
        solver = LS2SRSolver(graph=graph, args=args)

        solution = None
        dynamicity = np.zeros(shape=(te_step, 7))
        for i in tqdm(range(te_step)):
            mean = np.mean(y_gt[i], axis=1)
            std_mean = np.std(mean)
            std = np.std(y_gt[i], axis=1)
            std_std = np.std(std)

            maxmax_mean = np.max(y_gt[i]) / np.mean(y_gt[i])

            theo_lamda = calculate_lamda(y_gt=y_gt[i])

            pred_tm = yhat[i]
            u, solution = p2_heuristic_solver(solver, tm=pred_tm,
                                              gt_tms=y_gt[i], p_solution=solution, nNodes=args.nNodes)

            dynamicity[i] = [np.sum(y_gt[i]), std_mean, std_std, np.sum(std), maxmax_mean, np.mean(u), theo_lamda]

            _solution = np.copy(solution)
            results.append((u, _solution))

        mlu, solution = extract_results(results)
        route_changes = get_route_changes_heuristic(solution)

        print('Route changes: Total: {}  - Avg {:.3f} std {:.3f}'.format(np.sum(route_changes),
                                                                         np.sum(route_changes) /
                                                                         (args.seq_len_y * route_changes.shape[0]),
                                                                         np.std(route_changes)))
        print('gwn ls2sr    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                                np.min(mlu),
                                                                                np.mean(mlu),
                                                                                np.max(mlu),
                                                                                np.std(mlu)))
        congested = mlu[mlu >= 1.0].size
        print('Congestion_rate: {}/{}'.format(congested, mlu.size))

        save_results(args.log_dir, 'gwn_ls2sr_run_{}'.format(run_test), mlu, route_changes)
        np.save(os.path.join(args.log_dir, 'gwn_ls2sr_dyn_run_{}'.format(run_test)), dynamicity)


def createGraph_srls(NodesFile, EdgesFile):
    def addEdge(graph, src, dst, w, bw, delay, idx, capacity):
        graph.add_edge(src, dst, weight=w,
                       capacity=bw,
                       delay=delay)
        graph.edges[src, dst]['index'] = idx
        capacity.append(bw)

    capacity = []
    G = nx.DiGraph()
    df = pd.read_csv(NodesFile, delimiter=' ')
    for i, row in df.iterrows():
        G.add_node(i, label=row.label, pos=(row.x, row.y))

    nNodes = G.number_of_nodes()

    index = 0
    df = pd.read_csv(EdgesFile, delimiter=' ')
    for _, row in df.iterrows():
        i = row.src
        j = row.dest
        if (i, j) not in G.edges:
            addEdge(G, i, j, row.weight, row.bw, row.delay, index, capacity)
            index += 1
        if (j, i) not in G.edges:
            addEdge(G, j, i, row.weight, row.bw, row.delay, index, capacity)
            index += 1

    nEdges = G.number_of_edges()
    sPathNode = []
    sPathEdge = []
    nSPath = []
    for u in G.nodes:
        A = []
        B = []
        C = []
        for v in G.nodes:
            A.append(list(nx.all_shortest_paths(G, u, v)))
            B.append([])
            C.append(0)
            if len(A[-1][0]) >= 2:
                C[-1] = len(A[-1])
                for path in A[-1]:
                    B[-1].append([])
                    for j in range(len(path) - 1):
                        B[-1][-1].append(G[path[j]][path[j + 1]]['index'])
        sPathNode.append(A)
        sPathEdge.append(B)
        nSPath.append(C)
    capacity = CapacityData(capacity)
    sp = ShortestPaths(sPathNode, sPathEdge, nSPath)
    G.sp = sp
    return G, nNodes, nEdges, capacity, sp


def gwn_srls(yhat, y_gt, graphs, te_step, args):
    print('GWN SRLS')
    G, nNodes, nEdges, capacity, sp = graphs
    for run_test in range(args.nrun):

        results = []
        solver = SRLS(sp, capacity, nNodes, nEdges, args.timeout)
        LinkLoads, RoutingMatrices, TMs = [], [], []
        dynamicity = np.zeros(shape=(te_step, 7))
        for i in tqdm(range(te_step)):
            mean = np.mean(y_gt[i], axis=1)
            std_mean = np.std(mean)
            std = np.std(y_gt[i], axis=1)
            std_std = np.std(std)

            maxmax_mean = np.max(y_gt[i]) / np.mean(y_gt[i])

            theo_lamda = calculate_lamda(y_gt=y_gt[i])

            pred_tm = yhat[i]
            u, solutions, linkloads, routingMxs = p2_srls_solver(solver, tm=pred_tm, gt_tms=y_gt[i], nNodes=args.nNodes)
            solutions = np.asarray(solutions)
            dynamicity[i] = [np.sum(y_gt[i]), std_mean, std_std, np.sum(std), maxmax_mean, np.mean(u), theo_lamda]

            _solutions = np.copy(solutions)
            results.append((u, _solutions))
            LinkLoads.append(linkloads)
            RoutingMatrices.append(routingMxs)
            TMs.append(y_gt[i])

        LinkLoads = np.stack(LinkLoads, axis=0)
        RoutingMatrices = np.stack(RoutingMatrices, axis=0)
        TMs = np.stack(TMs, axis=0)

        mlu, solution = extract_results(results)
        route_changes = get_route_changes(solution, G)

        print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes) /
                                                            (args.seq_len_y * route_changes.shape[0]),
                                                            np.std(route_changes)))
        print('gwn SRLS    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                               np.min(mlu),
                                                                               np.mean(mlu),
                                                                               np.max(mlu),
                                                                               np.std(mlu)))
        congested = mlu[mlu >= 1.0].size
        print('Congestion_rate: {}/{}'.format(congested, mlu.size))

        save_results(args.log_dir, 'gwn_srls_run_{}'.format(run_test), mlu, route_changes)
        np.save(os.path.join(args.log_dir, 'gwn_srls_dyn'), dynamicity)

        # np.save(os.path.join(args.log_dir, 'LinkLoads_gwn_srls'), LinkLoads)
        # np.save(os.path.join(args.log_dir, 'RoutingMatrices_gwn_srls'), RoutingMatrices)
        # np.save(os.path.join(args.log_dir, 'TMs_gwn_srls'), TMs)


def gt_srls(y_gt, graphs, te_step, args):
    print('gt_srls')
    G, nNodes, nEdges, capacity, sp = graphs
    for run_test in range(args.nrun):

        results = []
        solver = SRLS(sp, capacity, nNodes, nEdges, args.timeout)
        LinkLoads, RoutingMatrices, TMs = [], [], []
        dynamicity = np.zeros(shape=(te_step, 7))
        for i in tqdm(range(te_step)):
            mean = np.mean(y_gt[i], axis=1)
            std_mean = np.std(mean)
            std = np.std(y_gt[i], axis=1)
            std_std = np.std(std)

            maxmax_mean = np.max(y_gt[i]) / np.mean(y_gt[i])

            theo_lamda = calculate_lamda(y_gt=y_gt[i])

            pred_tm = np.max(y_gt[i], axis=0, keepdims=True)
            u, solutions, linkloads, routingMxs = p2_srls_solver(solver, tm=pred_tm, gt_tms=y_gt[i], nNodes=args.nNodes)
            solutions = np.asarray(solutions)
            dynamicity[i] = [np.sum(y_gt[i]), std_mean, std_std, np.sum(std), maxmax_mean, np.mean(u), theo_lamda]

            _solutions = np.copy(solutions)
            results.append((u, _solutions))
            LinkLoads.append(linkloads)
            RoutingMatrices.append(routingMxs)
            TMs.append(y_gt[i])

        mlu, solution = extract_results(results)
        route_changes = get_route_changes(solution, G)
        LinkLoads = np.stack(LinkLoads, axis=0)
        RoutingMatrices = np.stack(RoutingMatrices, axis=0)
        TMs = np.stack(TMs, axis=0)

        print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes) /
                                                            (args.seq_len_y * route_changes.shape[0]),
                                                            np.std(route_changes)))
        print('gt srls     {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                               np.min(mlu),
                                                                               np.mean(mlu),
                                                                               np.max(mlu),
                                                                               np.std(mlu)))
        congested = mlu[mlu >= 1.0].size
        print('Congestion_rate: {}/{}'.format(congested, mlu.size))

        save_results(args.log_dir, 'gt_srls_run_{}'.format(run_test), mlu, route_changes)
        np.save(os.path.join(args.log_dir, 'gt_srls_dyn'), dynamicity)

        # np.save(os.path.join(args.log_dir, 'LinkLoads_gwn_srls'), LinkLoads)
        # np.save(os.path.join(args.log_dir, 'RoutingMatrices_gwn_srls'), RoutingMatrices)
        # np.save(os.path.join(args.log_dir, 'TMs_gwn_srls'), TMs)


def optimal_p0_solver(y_gt, G, segments, te_step, args):
    solver = OneStepSRSolver(G, segments)

    def f(gt_tms):
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms[gt_tms <= 0.0] = 0.0
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))

        return p0(solver, gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(gt_tms=y_gt[i]) for i in range(te_step))

    mlu, solution = extract_results(results)
    solution = np.reshape(solution, newshape=(-1, args.nNodes, args.nNodes, args.nNodes))
    rc = get_route_changes(solution, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('Optimal p0           | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                            np.mean(mlu),
                                                                            np.max(mlu),
                                                                            np.std(mlu)))
    congested = mlu[mlu >= 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p0', mlu, rc)


def optimal_p1_solver(y_gt, G, segments, te_step, args):
    solver = MultiStepSRSolver(G, segments)

    def f(gt_tms, tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))

        tms[tms <= 0.0] = 0.0
        gt_tms[gt_tms <= 0.0] = 0.0

        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))
        return p1(solver, tms, gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(
        tms=y_gt[i], gt_tms=y_gt[i]) for i in range(te_step))

    mlu, solution = extract_results(results)
    rc = get_route_changes(solution, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('Optimal p1               | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                                np.mean(mlu),
                                                                                np.max(mlu),
                                                                                np.std(mlu)))

    congested = mlu[mlu >= 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p1', mlu, rc)


def optimal_p2_solver(y_gt, G, segments, te_step, args):
    solver = MaxStepSRSolver(G, segments)

    def f(gt_tms, tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))

        tms[tms <= 0.0] = 0.0
        gt_tms[gt_tms <= 0.0] = 0.0

        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))
        tms = tms.reshape((args.nNodes, args.nNodes))

        return p2(solver, tms=tms, gt_tms=gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(
        gt_tms=y_gt[i], tms=np.max(y_gt[i], axis=0, keepdims=True)) for i in range(te_step))

    mlu, solution_optimal_p2 = extract_results(results)
    rc = get_route_changes(solution_optimal_p2, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('Optimal p2                | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                                 np.mean(mlu),
                                                                                 np.max(mlu),
                                                                                 np.std(mlu)))
    congested = mlu[mlu >= 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p2', mlu, rc)


def optimal_p3_solver(y_gt, G, segments, te_step, args):
    t_prime = int(args.seq_len_y / args.trunk)
    solver = MultiStepSRSolver(G, segments)

    def f(gt_tms, tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))

        tms[tms <= 0.0] = 0.0
        gt_tms[gt_tms <= 0.0] = 0.0

        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))

        return p3(solver, tms, gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(
        tms=np.stack([np.max(y_gt[i][j:j + t_prime], axis=0) for j in range(0, y_gt[i].shape[0], t_prime)]),
        gt_tms=y_gt[i]) for i in range(te_step))

    mlu, solution_optimal_p3 = extract_results(results)
    rc = get_route_changes(solution_optimal_p3, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('Optimal p3             | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                              np.mean(mlu),
                                                                              np.max(mlu),
                                                                              np.std(mlu)))
    congested = mlu[mlu >= 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p3', mlu, rc)


def p1(solver, tms, gt_tms):
    u = []
    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p3(solver, tms, gt_tms):
    u = []
    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p2(solver, tms, gt_tms):
    u = []

    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p0_ls2sr(solver, tms, gt_tms, p_solution, nNodes):
    u = []
    tms = tms.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tms = tms.reshape((-1, nNodes, nNodes))

    solutions = []
    for i in range(gt_tms.shape[0]):
        solution = solver.solve(tms[i], solution=p_solution)  # solve backtrack solution (line 131)
        u.append(solver.evaluate(solution, gt_tms[i]))
        solutions.append(solution)

    return u, solutions


def p2_heuristic_solver(solver, tm, gt_tms, p_solution, nNodes):
    u = []
    tm = tm.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tm[tm <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tm[:] = tm[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tm = tm.reshape((nNodes, nNodes))

    solution = solver.solve(tm, solution=p_solution)  # solve backtrack solution (line 131)

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(solution, gt_tms[i]))
    return u, solution


def flowidx2srcdst(flow_idx, nNodes):
    src = flow_idx / nNodes
    src = src.astype(np.int)

    dst = flow_idx % nNodes
    dst = dst.astype(np.int)

    srcdst_idx = np.stack([src, dst], axis=1)
    return srcdst_idx


def p2_cfr(solver, tm, gt_tms, pSolution, nNodes, num_cf):
    u = []

    tm = tm.flatten()
    topk_idx = np.argsort(tm)[::-1]
    topk_idx = topk_idx[:num_cf]

    rTm = np.copy(tm)
    rTm[topk_idx] = 0
    rTm = rTm.reshape((nNodes, nNodes))

    tm = tm[topk_idx]

    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))
    gt_tms[gt_tms <= 0.0] = 0.0
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))

    srcdst_idx = flowidx2srcdst(flow_idx=topk_idx, nNodes=nNodes)
    solution = solver.solve(tm=tm, rTm=rTm, flow_idx=srcdst_idx, pSolution=pSolution)

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(solution=solution, tm=gt_tms[i]))
    return u, solution


def p2_srls_solver(solver, tm, gt_tms, nNodes):
    u = []
    tm = tm.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tm[tm <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tm[:] = tm[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tm = tm.reshape((nNodes, nNodes))

    try:
        solver.modifierTrafficMatrix(tm)  # solve backtrack solution (line 131)
        solver.solve()
    except:
        print('ERROR in p2_srls_solver --> pass')
        pass

    solution = solver.extractRoutingPath()
    linkloads, routingMxs = [], []
    solutions = []

    for i in range(gt_tms.shape[0]):
        solutions.append(solution)

        u.append(solver.evaluate(solution, gt_tms[i]))

        linkload = solver.getLinkload(routingSolution=solution, trafficMatrix=gt_tms[i])
        routingMx = solver.getRoutingMatrix(routingSolution=solution)
        linkloads.append(linkload)
        routingMxs.append(routingMx)

    return u, solutions, linkloads, routingMxs


def p2_srls_fix_max_solver(solver, solution, gt_tms, nNodes):
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))
    gt_tms[gt_tms <= 0.0] = 0.0
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))

    linkloads, routingMxs = [], []

    for i in range(gt_tms.shape[0]):
        linkload = solver.getLinkload(routingSolution=solution, trafficMatrix=gt_tms[i])
        routingMx = solver.getRoutingMatrix(routingSolution=solution)
        linkloads.append(linkload)
        routingMxs.append(routingMx)

    return linkloads, routingMxs


def p0_srls_solver(solver, tms, gt_tms, nNodes):
    u = []
    tms = tms.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tms = tms.reshape((-1, nNodes, nNodes))

    solutions = []
    linkloads, routingMxs = [], []
    for i in range(gt_tms.shape[0]):
        try:
            solver.modifierTrafficMatrix(tms[i])  # solve backtrack solution (line 131)
            solver.solve()
        except:
            print('ERROR in p2_srls_solver --> pass')
            pass
        solution = solver.extractRoutingPath()

        u.append(solver.evaluate(solution, gt_tms[i]))
        solutions.append(solution)

        linkload = solver.getLinkload(routingSolution=solution, trafficMatrix=gt_tms[i])
        routingMx = solver.getRoutingMatrix(routingSolution=solution)
        linkloads.append(linkload)
        routingMxs.append(routingMx)

    return u, solutions, linkloads, routingMxs


def last_step_sr(solver, last_tm, gt_tms):
    u = []
    try:
        solver.solve(last_tm)
    except:
        pass

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def first_step_sr(solver, first_tm, gt_tms):
    u = []
    try:
        solver.solve(first_tm)
    except:
        pass

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def one_step_predicted_sr(solver, tm, gt_tms):
    u = []
    try:
        solver.solve(tm)
    except:
        pass

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p0(solver, gt_tms):
    u = []
    solutions = []
    for i in range(gt_tms.shape[0]):
        try:
            solver.solve(gt_tms[i])
        except:
            pass

        solution = solver.solution
        solutions.append(solution)
        u.append(solver.evaluate(gt_tms[i], solution))

    solutions = np.stack(solutions, axis=0)
    return u, solutions


def oblivious_sr(solver, tms):
    u = []
    for i in range(tms.shape[0]):
        u.append(solver.evaluate(tms[i]))

    return u, solver.solution


def run_te(x_gt, y_gt, yhat, args):
    print('|--- run TE on DIRECTED graph')

    te_step = x_gt.shape[0]
    print('    Method           |   Min     Avg    Max     std')

    if args.run_te == 'gwn_ls2sr':
        graph = load_network_topology(args.dataset, args.datapath)
        gwn_ls2sr(yhat, y_gt, graph, te_step, args)
    elif args.run_te == 'gwn_srls':
        graphs = createGraph_srls(os.path.join(args.datapath, 'topo/{}_node.csv'.format(args.dataset)),
                                  os.path.join(args.datapath, 'topo/{}_edge.csv'.format(args.dataset)))
        gwn_srls(yhat, y_gt, graphs, te_step, args)
    elif args.run_te == 'gt_srls':
        graphs = createGraph_srls(os.path.join(args.datapath, 'topo/{}_node.csv'.format(args.dataset)),
                                  os.path.join(args.datapath, 'topo/{}_edge.csv'.format(args.dataset)))
        gt_srls(y_gt, graphs, te_step, args)
    elif args.run_te == 'p0':
        graph = load_network_topology(args.dataset, args.datapath)
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p0_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'p1':
        graph = load_network_topology(args.dataset, args.datapath)
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p1_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'p2':  # (or gt_p2)
        graph = load_network_topology(args.dataset, args.datapath)
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p2_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'p3':
        graph = load_network_topology(args.dataset, args.datapath)
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p3_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'or':
        graph = load_network_topology(args.dataset, args.datapath)
        segments = compute_path(graph, args.dataset, args.datapath)
        oblivious_routing_solver(y_gt, graph, segments, te_step, args)
    else:
        raise RuntimeError('TE not found!')
