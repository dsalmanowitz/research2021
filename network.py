import matplotlib.pyplot as plt
import networkx as nx
import random
import copy

def r():
    return random.random()

def get_graph():
    G = nx.DiGraph()
    G.add_nodes_from([
        (0, {"inf": "y", "thresh": r()}),
        (1, {"inf": "n", "thresh": r()}),
        (2, {"inf": "n", "thresh": r()}),
        (3, {"inf": "n", "thresh": r()}),
        (4, {"inf": "n", "thresh": r()}),
    ])
    G.add_weighted_edges_from([
        (0,1,r()), (0,2,r()), (0,3,r()), (0,4,r()), 
        (1,2,r()), (1,3,r()), (1,4,r()),
        (2,3,r()), (2,4,r()),
        (3,4,r())
    ])
    return G

def show(G):
    colors = []
    for i in G.nodes:
        colors.append("lightcoral" if G.nodes[i]["inf"] == "n" else "yellowgreen")
    nx.draw(G, with_labels=True, node_color=colors)
    plt.show()

def cascade_step(G, active):
    new = []
    for i in G.nodes:
        if G.nodes[i]["inf"] == "y" and i in active:
            suc = list(G.successors(i))
            for j in suc:
                if G.nodes[j]["inf"] == "n" and G[i][j]["weight"] > G.nodes[j]["thresh"]:
                    G.nodes[j]["inf"] = "y"
                    new.append(j)
    return G, new

def cascade(G, active):
    graphs = [G]
    newNodes = [active]
    while True:
        nxt, new = cascade_step(copy.deepcopy(graphs[-1]), newNodes[-1])
        if new == []:
            break
        newNodes.append(new)
        graphs.append(nxt)
    return graphs, newNodes

def lin_thresh_step(G):
    new = []
    for i in G.nodes:
        if G.nodes[i]["inf"] == "n":
            pred = list(G.predecessors(i))
            sum = 0
            for j in pred:
                sum += G[j][i]["weight"]
            if sum > G.nodes[i]["thresh"]:
                G.nodes[i]["inf"] = "y"
                new.append(i)
    return G, new

def lin_thresh(G):
    graphs = [G]
    newNodes = [[0]]
    while True:
        nxt, new = lin_thresh_step(copy.deepcopy(graphs[-1]))
        if new == []:
            break
        graphs.append(nxt)
        newNodes.append(new)
    return graphs, newNodes
