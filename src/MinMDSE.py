import pickle
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import xml.etree.cElementTree as et
from joblib import Parallel, delayed
from shapely.geometry import Point, MultiLineString, Polygon
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

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

def doWhile(G, NodeA, weight=None):
    count = 0
    L = len(NodeA)
    for Xi in NodeA:
        NodeB = NodeA.copy()
        NodeB.remove(Xi)
        for Xj in NodeB:
            delt_ij = delt_Xij(Xi, Xj, G, weight=weight)  # ??????
            if delt_ij >= 0:
                return True
    return False

def delt_Xij(Xi, Xj, G, weight=None):
    Xij = Xi + Xj
    sub_set = [Xi, Xj, list(set(Xij))]
    degree = G.degree(weight=weight)
    G_volume = sum(degree[index_node] for index_node in G.nodes)
    Vij, gij = [], []
    for ind in range(len(sub_set)):
        sub_degree = 0
        for node in sub_set[ind]:
            sub_degree += degree[node]
        gj_c = nx.cut_size(G, sub_set[ind], weight=weight)
        Vij.append(sub_degree)
        gij.append(gj_c)
    gij = np.array(gij)
    Vij = np.array(Vij)
    g_i, g_j, g_ij = gij[0], gij[1], gij[2]
    V_i, V_j, V_ij = Vij[0], Vij[1], Vij[2]
    log_Vij = Log2p(Vij, eps=1e-10)
    delt_G_Pij = 1.0 / (G_volume) * ((V_i - g_i) * log_Vij[0] +
                                     (V_j - g_j) * log_Vij[1] -
                                     (V_ij - g_ij) * log_Vij[2] +
                                     (g_i + g_j - g_ij) * np.log2(G_volume + 1e-10))
    return delt_G_Pij

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

def get_oneStruInforEntropy(G, weight=None):
    G_du = G.degree(weight=weight)
    G_volume = sum(G_du[index_node] for index_node in G.nodes)
    G_pu_dic = {index_node: G_du[index_node] * 1.0 / (1.0 * G_volume) for index_node in G.nodes}
    G_pu = [G_pu_dic[index_node] for index_node in G.nodes]
    # Shonnon Entropy
    HP_G_Shonnon = sum(pLog2p(np.array(G_pu))) * (-1.0)
    return HP_G_Shonnon

def min2DStruInforEntropyPartition(G, weight=None, pk_partion=None):
    # Input ???????????? -- ??????
    print("Partition by min2DHG ..........")
    nodes = list(G.nodes())
    nodes.sort()  # ????????????????????????
    global NodeA
    if pk_partion is None:
        nodes = list(G.nodes())
        nodes.sort()  # ????????????????????????
        NodeA = [[node] for node in nodes]
    else:
        NodeA = pk_partion
    print("Init-Input:", NodeA)  # Input Data
    doWhileFlg = True
    NodeA.reverse()  # ??????
    while doWhileFlg:
        Xi = NodeA.pop()
        Nj = NodeA.copy()
        delt_max = 0
        Xj_m = None
        for Xj in Nj:
            delt_ij = delt_Xij(Xi, Xj, G, weight=weight)  # ??????
            if delt_ij > 0 and delt_ij > delt_max:
                Xj_m = Xj
                delt_max = delt_ij
        if Xj_m in Nj and Xj_m is not None:
            Nj.remove(Xj_m)
            Xij = Xi + Xj_m
            Nj.insert(0, Xij)
            # print('Xi:', Xi, '+ Xj:', Xj_m, '-->', Xij, ' delt_ij_HG:', delt_max)
        elif Xj_m is None:
            Nj.insert(0, Xi)  # ???????????????
        NodeA = Nj
        doWhileFlg = doWhile(G, NodeA, weight=weight)  # ??????????????????
        # print(NodeA)
    # print('Output:', NodeA)
    sub_set = NodeA.copy()  # Final Result
    # Output: NodeA ???????????? -- ??????

    # sort
    results = []
    for sb_result in sub_set:
        sb_result.sort()
        results.append(sb_result)
    results.sort()
    print('Output:', results)
    return results

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

def show_graph(G, weight='weight', with_labels=False, save=False, filename='./graph.svg'):
    # test: outer point
    print("G: number_of_nodes:{0}, number_of_edges:{1}".format(G.number_of_nodes(), G.number_of_edges()))
    # pos = nx.spring_layout(G)  # set layout
    pos = nx.kamada_kawai_layout(G)
    # nx.draw(G, pos=pos, node_size=300, with_labels=with_labels, node_color='red')
    nodecolor = G.degree(weight=weight)  # ???????????????????????????????????????????????????
    nodecolor2 = pd.DataFrame(nodecolor)  # ?????????????????????
    nodecolor3 = nodecolor2.iloc[:, 1]  # ???????????????
    edgecolor = range(G.number_of_edges())  # ??????????????????

    nx.draw(G,
            pos,
            with_labels=with_labels,
            node_size=nodecolor3 * 6 * 10,
            node_color=nodecolor3 * 5,
            edge_color=edgecolor)

    if save:
        plt.savefig(filename, dpi=600, transparent=True)
    plt.show()

def test_StruInforEntropy():
    # ???????????? or all weights == 1.0
    with open('../data/cora/cora.cites', 'r') as f:
        adj_list = {}
        v = set()
        for line in f.readlines():
            cited, citing = line.split('\t')
            cited, citing = eval(cited), eval(citing)
            if cited in adj_list.keys():
                adj_list[cited].append(citing)
            else:
                adj_list[cited] = [citing]
            v.update([cited, citing])

    # ????????????
    v = list(v)
    # eye????????????????????????
    adj_mtrx = np.eye(len(v), dtype='int32')

    text = str()

    for cited in adj_list.keys():
        for citing in adj_list[cited]:
            adj_mtrx[v.index(cited), v.index(citing)] = 1
            adj_mtrx[v.index(citing), v.index(cited)] = 1
            text += "(" + str(v.index(cited)) + ", " + str(v.index(citing)) + ", {'weight': 1.0}),"

    text = "[" + text.strip(',') + "]"
    edges_list = eval(text)

    G = nx.Graph()
    G.add_edges_from(edges_list)

    # plot graph
    show_graph(G, with_labels=True)

    pk_results = []
    # min2DStruInforEntropyPartition
    results = min2DStruInforEntropyPartition(G, weight='weight', pk_partion= pk_results)
    print("Partition-Size(by min2DHG):", len(results))
    print("Partition(by min2DHG):", results)
    # 2D-StruInforEntropy
    HG, Vj, g_j, Gj_deg = StruInforEntropy(G, partition=results, weight='weight')
    oneHG = get_oneStruInforEntropy(G, weight='weight')
    print("1DStruInforEntropy:", oneHG)
    print("2DStruInforEntropy:", HG)
    # get_coo_matrix_from_G
    row, col, data = get_coo_matrix_from_G(G, weight='weight')
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            ndq_a, ndq_b = results[i], results[j]
            DI_ij = deltDI_ij(row, col, data, ndq_a=ndq_a, ndq_b=ndq_b)
            DI_ij = float("%.3f" % (DI_ij))
            # print(ndq_a, '+', ndq_b, '-', ndq_a + ndq_b, 'deltDI=', DI_ij)
    print('--'*15)


def main():
    # test_StruInforEntropy
    test_StruInforEntropy()

if __name__ == '__main__':
    main()
