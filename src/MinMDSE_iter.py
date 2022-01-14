import numpy as np
from joblib import Parallel, delayed
import networkx as nx
import six.moves.cPickle as pkl

def get_coo_matrix_from_G(G, weight=None):
    from scipy.sparse import coo_matrix
    row = np.array([u for u, _ in G.edges(data=False)])
    col = np.array([v for _, v in G.edges(data=False)])
    if weight is not None:
        data = np.array([w['weight'] for _, __, w in G.edges(data=True)])
    else:
        data = np.ones(shape=row.shape, dtype=np.float)
    coo_mat = coo_matrix((data, (row, col)), dtype=np.float)
    row = coo_mat.row
    col = coo_mat.col
    data = coo_mat.data
    return row, col, data

def parts_DI(row, col, data, ndq):
    deg_ndq = {}  # ndq degrees
    nodes = []
    weights = []
    for nd in ndq:
        index_row = np.where(row == nd)
        index_col = np.where(col == nd)
        u = list(row[index_col]) + list(col[index_row])
        index_data = np.array(list(index_row[0]) + list(index_col[0]), dtype=np.int)
        w = list(data[index_data])
        deg_ndq[nd] = np.sum(w)
        nodes += u
        weights += w
    for nd in ndq:
        for _ in range(nodes.count(nd)):
            nodes.remove(nd)
    V_G = np.sum(data) * 2.0
    # deg_i = deg_ndq
    gi = len(nodes)
    Vi = np.sum(weights)
    V_div = Vi / V_G
    V_div_hat = (Vi - gi) / V_G

    return 0.0 if V_div < 1e-5 else -V_div_hat * np.log2(V_div)

def partition_producer(partition):
    for L in partition:
        yield L

def deltDI_ij(row, col, data, ndq_a, ndq_b):
    ndq = ndq_a + ndq_b
    L_X, L_Y, L_XY = len(ndq_a), len(ndq_b), len(ndq)
    deg_ndq = {}  # ndq degrees
    nodes_a, weights_a = [], []
    nodes_b, weights_b = [], []
    nodes, weights = [], []
    for nd in ndq[:L_X]:
        index_row = np.where(row == nd)
        index_col = np.where(col == nd)
        u = list(row[index_col]) + list(col[index_row])
        index_data = np.array(list(index_row[0]) + list(index_col[0]), dtype=np.int)
        w = list(data[index_data])
        deg_ndq[nd] = np.sum(w)
        nodes += u
        weights += w
    nodes_a += nodes
    weights_a += weights
    for nd in ndq[L_X:]:
        index_row = np.where(row == nd)
        index_col = np.where(col == nd)
        u = list(row[index_col]) + list(col[index_row])
        index_data = np.array(list(index_row[0]) + list(index_col[0]), dtype=np.int)
        w = list(data[index_data])
        deg_ndq[nd] = np.sum(w)
        nodes += u
        weights += w
    nodes_b += nodes[len(nodes_a):len(nodes)]
    weights_b += weights[len(weights_a):len(weights)]

    for nd in ndq[:L_X]:
        for _ in range(nodes.count(nd)):
            nodes.remove(nd)
        for _ in range(nodes_a.count(nd)):
            nodes_a.remove(nd)
    for nd in ndq[L_X:]:
        for _ in range(nodes.count(nd)):
            nodes.remove(nd)
        for _ in range(nodes_b.count(nd)):
            nodes_b.remove(nd)
    # deg_ij = deg_ndq
    # deg_i = {nd: deg_ndq[nd] for nd in ndq_a}
    # deg_j = {nd: deg_ndq[nd] for nd in ndq_b}
    V_G = np.sum(data) * 2.0
    g_i = len(nodes_a)
    V_i = np.sum(weights_a)
    g_j = len(nodes_b)
    V_j = np.sum(weights_b)
    g_ij = len(nodes)
    V_ij = np.sum(weights)

    if V_i < 1e-5:
        V_i = 1.0
    if V_j < 1e-5:
        V_j = 1.0
    if V_ij < 1e-5:
        V_ij = 1.0
    if V_G < 1e-5:
        V_G = 1.0

    delt = -(V_i - g_i) * np.log2(V_i) - (V_j - g_j) * np.log2(V_j) \
           + (V_ij - g_ij) * np.log2(V_ij) \
           + (V_i + V_j - V_ij - g_i - g_j + g_ij) * np.log2(V_G)

    return delt / V_G

def producer(N):
    ndi = N[0]
    N.remove(ndi)
    for ndj in N:
        yield list(ndi), list(ndj)

def iter_maxDI(G, iter_max=100, pk_partion=None, weight=None, n_jobs=4, verbose=0):
    # iter_max = 100
    # n_jobs = 4
    print("Partition by iter_maxDI ..........")
    row, col, data = get_coo_matrix_from_G(G, weight=weight)
    global N
    if pk_partion is None:
        nodes = list(G.nodes())
        nodes.sort()  # 节点编号升序排列
        N = [[node] for node in nodes]
    else:
        N = pk_partion
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(parts_DI)(row, col, data, P) for P in partition_producer(N))
    # print('(iter:%d ---> DI:%.2f bits)' % (0, np.sum(out)))
    DI = np.zeros(iter_max + 1)
    DI[0] = float('%.3f' % np.sum(out))
    print('(iter:%d ---> DI:%.3f bits)' % (0, DI[0]))
    for iter in range(iter_max):
        ndi = N[0]
        out = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(deltDI_ij)(row, col, data, ndi, ndj) for ndi, ndj in producer(N))
        out_min = min(out)
        if out_min < 0:
            ndj = N[out.index(out_min)]
            N.remove(ndj)
            N.append(ndi + ndj)
        elif ndi not in N:
            N.append(ndi)
        out = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(parts_DI)(row, col, data, P) for P in partition_producer(N))
        DI[iter + 1] = float('%.3f' % np.sum(out))
        if (iter + 1) % 10 == 0:
            print('(iter:%d ---> DI:%.3f bits)' % (iter+1, DI[iter+1]))
        # if (iter + 1) >= 2000 and np.var(DI[-2000:-1], ddof=1) < 1e-10:  # 计算样本方差 （ 计算时除以 N - 1 ）
        #     DI = DI[:iter + 2]
        #     break

    # sort results
    results = []
    for sb_result in N:
        sb_result.sort()
        results.append(sb_result)
    results.sort()

    return results, DI

def StruInforEntropy(G, partition, weight=None):
    # nodes = G.nodes
    # n = G.number_of_nodes()
    # m = G.number_of_edges()
    sub_set = partition.copy()
    degree = G.degree(weight=weight)
    G_volume = sum(degree[index_node] for index_node in G.nodes)
    Vij, gij, deg_ij = [], [], []
    for ind in range(len(sub_set)):
        sub_degree = 0
        dij = []
        for node in sub_set[ind]:
            sub_degree += degree[node]
            dij.append(degree[node])
        gj_c = nx.cut_size(G, sub_set[ind], weight=weight)
        Vij.append(sub_degree)
        gij.append(gj_c)
        deg_ij.append(np.array(dij))
    gij = np.array(gij, dtype=float)
    Vij = np.array(Vij, dtype=float)
    p_i = [deg_ij[i] / Vij[i] for i in range(len(Vij))]
    pLogp = [pLog2p(pi, eps=1e-10) for pi in p_i]
    sum_pLogp = np.array([np.sum(plp) for plp in pLogp], dtype=float)

    first = np.sum((-1.0) * Vij / (G_volume) * sum_pLogp)
    second = np.sum((-1.0) * gij / (G_volume) * Log2p(Vij / (G_volume)))

    HG = first + second

    return HG, Vij, gij, deg_ij

def pLog2p(p_i, eps=1e-10):
    ind = np.where(p_i < eps)
    if len(ind) > 0:
        p_i[p_i < eps] = 1.0
        return p_i * np.log2(p_i)
    else:
        return p_i * np.log2(p_i)

def Log2p(p_i, eps=1e-10):
    ind = np.where(p_i < eps)
    if len(ind) > 0:
        p_i[p_i < eps] = 1.0
        return np.log2(p_i)
    else:
        return np.log2(p_i)

def get_oneStruInforEntropy(G, weight=None):
    G_du = G.degree(weight=weight)
    G_volume = sum(G_du[index_node] for index_node in G.nodes)
    G_pu_dic = {index_node: G_du[index_node] * 1.0 / (1.0 * G_volume) for index_node in G.nodes}
    G_pu = [G_pu_dic[index_node] for index_node in G.nodes]
    # Shonnon Entropy
    HP_G_Shonnon = sum(pLog2p(np.array(G_pu))) * (-1.0)
    return HP_G_Shonnon

def main():
    G = nx.Graph()
    path = "../data/Pubmed/raw/"
    dataset = "pubmed"
    adj_list = pkl.load(open('{}ind.{}.graph'.format(path, dataset), 'rb'), encoding='latin1')
    v = list(set(adj_list.keys()))
    adj_mtrx = np.eye(len(v), dtype='int32')
    text = str()

    for cited in adj_list.keys():
        for citing in adj_list[cited]:
            adj_mtrx[v.index(cited), v.index(citing)] = 1
            adj_mtrx[v.index(citing), v.index(cited)] = 1
            text += "(" + str(v.index(cited)) + ", " + str(v.index(citing)) + ", {'weight': 1.0}),"

    text = "[" + text.strip(',') + "]"
    # print(text)
    edges_list = eval(text)
    G.add_edges_from(edges_list)

    pk_partion = []
    results_, DI_ = iter_maxDI(G, iter_max=1000, pk_partion=pk_partion, weight='weight', n_jobs=-1, verbose=0)
    print(results_, DI_)
    print("Partition-Size(by min2DHG):", len(results_))
    print("Partition(by min2DHG):", results_)
    # 2D-StruInforEntropy
    HG, Vj, g_j, Gj_deg = StruInforEntropy(G, partition=results_, weight='weight')
    oneHG = get_oneStruInforEntropy(G, weight='weight')
    print("1DStruInforEntropy:", oneHG)
    print("2DStruInforEntropy:", HG)
    # get_coo_matrix_from_G
    row, col, data = get_coo_matrix_from_G(G, weight='weight')
    for i in range(len(results_)):
        for j in range(i + 1, len(results_)):
            ndq_a, ndq_b = results_[i], results_[j]
            DI_ij = deltDI_ij(row, col, data, ndq_a=ndq_a, ndq_b=ndq_b)
            DI_ij = float("%.3f" % (DI_ij))
            # print(ndq_a, '+', ndq_b, '-', ndq_a + ndq_b, 'deltDI=', DI_ij)
    print('--' * 15)

if __name__ == '__main__':
    main()