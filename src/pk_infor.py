import pickle
import numpy as np
import networkx as nx
import infomap
from sklearn import metrics
from src.utils_SI import plot_graph, get_DI_Parallel, label_result, iter_maxDI
from src.matrix2knn_graph import getRadiusGraph
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def findPrioriKnowledges(G, initial_partition=None, edge_data_flg=False):
    infomapX = infomap.Infomap("--two-level")

    print("Building Infomap network from a NetworkX graph...")
    # for e in G.edges(data=edge_data_flg):
    #     infomapX.addLink(*e)
    if edge_data_flg:
        for v, u, w in G.edges(data=edge_data_flg):
            e = (v, u, w['weight'])
            infomapX.addLink(*e)
    else:
        for e in G.edges(data=edge_data_flg):
            infomapX.addLink(*e)

    print("Find communities with Infomap...")
    # print(infomap)
    # infomapX.run(initial_partition=initial_partition, silent=True)
    print("Found {} modules with codelength: {}".format(infomapX.numTopModules(), infomapX.codelength))
    communities = {}
    for node in infomapX.iterLeafNodes():
        communities[node.physicalId] = node.moduleIndex()
    nx.set_node_attributes(G, values=communities, name='community')
    return infomapX.numTopModules(), communities

def drawGraph(G):
    # position map
    # pos = nx.spring_layout(G)
    pos = nx.kamada_kawai_layout(G)
    # community ids
    communities = [v for k, v in nx.get_node_attributes(G, 'community').items()]
    numCommunities = max(communities) + 1
    # color map from http://colorbrewer2.org/
    cmapLight = colors.ListedColormap(['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6'], 'indexed',
                                      numCommunities)
    cmapDark = colors.ListedColormap(['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a'], 'indexed', numCommunities)
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    # Draw nodes
    nodeCollection = nx.draw_networkx_nodes(G,
                                            pos=pos,
                                            node_color=communities,
                                            cmap=cmapLight)
    # Set node border color to the darker shade
    darkColors = [cmapDark(v) for v in communities]
    nodeCollection.set_edgecolor(darkColors)
    # Draw node labels
    for n in G.nodes():
        plt.annotate(n,
                     xy=pos[n],
                     textcoords='offset points',
                     horizontalalignment='center',
                     verticalalignment='center',
                     xytext=[0, 0],
                     color=cmapDark(communities[n]))
    plt.axis('off')
    # plt.savefig("karate.png")
    plt.show()


def getGraphPKL(pkname='G.pkl', show=False):
    output = open(pkname, 'rb')
    G = pickle.load(output)
    output.close()
    if show:
        plot_graph(G, weight=None, with_labels=False, save=False, filename='./graph.svg')
    return G


def getXPKL(pkname='X.pkl'):
    output = open(pkname, 'rb')
    X = pickle.load(output)
    output.close()
    return X

def get_PKResult(G, initial_partition=None, gshow=False, edge_data_flg=False):
    print("Partition by Infomap ..........")
    numModules, pks = findPrioriKnowledges(G, initial_partition=initial_partition, edge_data_flg=edge_data_flg)
    if gshow:
        drawGraph(G)
    list_values = [val for val in pks.values()]
    print(list_values)
    list_nodes = [node for node in pks.keys()]
    # print('pk_result sortting .......')
    result = []
    for ndpa in range(numModules):
        nodes_part_index = np.argwhere(np.array(list_values) == ndpa).ravel()
        nodes_part = list(np.array(list_nodes)[nodes_part_index])
        nodes_part.sort()
        result.append(nodes_part)
    result.sort()

    output = open('info_pk_partion.pkl', 'wb')
    pickle.dump(result, output)
    output.close()

    return result

########################## Test #################################
def main():
    G = nx.Graph()
    with open('../data/citeseer/citeseer.cites', 'r') as f:
        adj_list = {}
        v = set()
        for line in f.readlines():
            cited, citing = line.split('\t')
            # citesser 要把这句话去掉
            # cited, citing = eval(cited), eval(citing)
            if cited in adj_list.keys():
                adj_list[cited].append(citing)
            else:
                adj_list[cited] = [citing]
            v.update([cited, citing])

    # 节点编号
    v = list(v)
    # eye函数创建对角矩阵
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
    # plot_graph(G)
    # # Priori Knowledge --->  get_PK_Result
    result = get_PKResult(G, initial_partition=None, gshow=False, edge_data_flg=False)
    print("result:",result)

    # DI
    DI = get_DI_Parallel(G, partion=result, weight=None, n_jobs=-1, verbose=0)
    print("DI:", DI)

    print('DI iter ......')
    results, DI = iter_maxDI(G, iter_max=10000, pk_partion=None, weight='weight', n_jobs=4, verbose=0)  # 并行加速
    print("results:", results)
    print(len(results))
    print(DI[-1])
    print("DI:", DI)

    output = open('results.pkl', 'wb')
    pickle.dump(results, output)
    output.close()
    output = open('DI.pkl', 'wb')
    pickle.dump(DI, output)
    output.close()

if __name__ == '__main__':
    main()