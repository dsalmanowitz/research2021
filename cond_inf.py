import matplotlib.pyplot as plt
import networkx as nx
import random
import copy

def r():
    return random.random()

def get_graph():
    G = nx.DiGraph()
    G.add_nodes_from([
        (0, {"inf": 3}),
        (1, {"inf": 0}),
        (2, {"inf": 0}),
        (3, {"inf": 0}),
        (4, {"inf": 0}),
        (5, {"inf": 0}),
        (6, {"inf": 0}),
        (7, {"inf": 0}),
        (8, {"inf": 0}),
        (9, {"inf": 0}),
        (10, {"inf": 0}),
        (11, {"inf": 0}),
        (12, {"inf": 0}),
        (13, {"inf": 0}),
        (14, {"inf": 0}),
        (15, {"inf": 0})
    ])
    G.add_weighted_edges_from([
        (0,1,0), (0,2,0), (0,3,0), (0,4,0), 
        (1,5,0), (1,6,0), (1,8,0),
        (2,5,0), (2,6,0), (2,7,0),
        (3,6,0), (3,7,0),
        (4,6,0), (4,7,0), (4,11,0),
        (5,8,0), (5,9,0),
        (6,9,0), (6,10,0),
        (7,10,0), (7,11,0),
        (8,12,0), (8,13,0),
        (9,12,0), (9,13,0),
        (10,13,0), (10,14,0),
        (11,13,0), (11,14,0),
        (12,15,0),
        (13,15,0),
        (14,15,0)
    ])
    return G

def show(G):
    colors = []
    for i in G.nodes:
        inf = G.nodes[i]["inf"]
        if inf == 3: colors.append("#008000")
        elif inf == 2: colors.append("#FFFF00")
        elif inf == 1: colors.append("#FFC0CB")
        else: colors.append("#FF0000")
    nx.draw(G, with_labels=True, node_color=colors)
    plt.show()

def set_edge_weight(G, i, j):
    x = r()
    if G.nodes[i]["inf"] == 3:
        if x < 0.65: G[i][j]["weight"] = 3
        elif 0.65 <= x < 0.85: G[i][j]["weight"] = 2
        elif 0.85 <= x < 0.95: G[i][j]["weight"] = 1
        else: G[i][j]["weight"] = 0
    elif G.nodes[i]["inf"] == 2:
        if x < 0.5: G[i][j]["weight"] = 2
        elif 0.5 <= x < 0.9: G[i][j]["weight"] = 1
        else: G[i][j]["weight"] = 0
    elif G.nodes[i]["inf"] == 1:
        if x < 0.5: G[i][j]["weight"] = 1
        else: G[i][j]["weight"] = 0
    else:
        G[i][j]["weight"] = 0

def max_inf(G):
    graphs = [G]
    newNodes = [0]
    while len(newNodes) != 16:
        nxt = max_inf_step(copy.deepcopy(graphs[-1]), newNodes)
        graphs.append(nxt)
    return graphs, newNodes

def max_inf_step(G, new):
    step_list = []
    for i in G.nodes:
        if i not in new:
            pred = list(G.predecessors(i))
            max_weight = 0
            for j in pred:
                if j in step_list:
                    return G
                set_edge_weight(G,j,i)
                if G[j][i]["weight"] > max_weight:
                    max_weight = G[j][i]["weight"]
            G.nodes[i]["inf"] = max_weight
            new.append(i)
            step_list.append(i)
    return G

def min_inf(G):
    graphs = [G]
    newNodes = [0]
    while len(newNodes) != 16:
        nxt = min_inf_step(copy.deepcopy(graphs[-1]), newNodes)
        graphs.append(nxt)
    return graphs, newNodes

def min_inf_step(G, new):
    step_list = []
    for i in G.nodes:
        if i not in new:
            pred = list(G.predecessors(i))
            min_weight = 3
            for j in pred:
                if j in step_list:
                    return G
                set_edge_weight(G,j,i)
                if G[j][i]["weight"] < min_weight:
                    min_weight = G[j][i]["weight"]
            G.nodes[i]["inf"] = min_weight
            new.append(i)
            step_list.append(i)
    return G

def med_inf(G):
    graphs = [G]
    newNodes = [0]
    while len(newNodes) != 16:
        nxt = med_inf_step(copy.deepcopy(graphs[-1]), newNodes)
        graphs.append(nxt)
    return graphs, newNodes

def med_inf_step(G, new):
    step_list = []
    for i in G.nodes:
        if i not in new:
            pred = list(G.predecessors(i))
            for j in pred:
                if j in step_list:
                    return G
                set_edge_weight(G,j,i)
            if len(pred) % 2 == 0:
                G.nodes[i]["inf"] = (G[pred[len(pred)//2]][i]["weight"]+G[pred[len(pred)//2-1]][i]["weight"])//2
            else:
                G.nodes[i]["inf"] = G[pred[len(pred)//2]][i]["weight"]
            new.append(i)
            step_list.append(i)
    return G

