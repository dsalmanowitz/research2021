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

def get_graph1(n=16, p=0.3):
    G = nx.fast_gnp_random_graph(n, p, directed=True)
    for i in G.nodes:
        G.nodes[i]["inf"] = 0
    G.nodes[0]["inf"] = 3
    G.nodes[1]["inf"] = -3
    for (start, end) in G.edges:
        G.edges[start, end]["weight"] = 0
    return G

def show(G):
    if type(G) != list:
        G = [G]
    for j in range(len(G)):
        plot = plt.figure(j+1)
        colors = []
        for i in G[j].nodes:
            inf = G[j].nodes[i]["inf"]
            if inf == 3: colors.append("#008000")
            elif inf == 2: colors.append("#FFFF00")
            elif inf == 1: colors.append("#FFC0CB")
            else: colors.append("#FF0000")
        nx.draw(G[j], with_labels=True, node_color=colors)
    plt.show()
    
def show1(G):
    if type(G) != list:
        G = [G]
    for j in range(len(G)):
        plot = plt.figure(j+1)
        colors = []
        for i in G[j].nodes:
            inf = G[j].nodes[i]["inf"]
            if inf == 3: colors.append("#4D8C57")
            elif inf == 2: colors.append("#78A161")
            elif inf == 1: colors.append("#A3B56B")
            elif inf == 0: colors.append("#FFFF00")
            elif inf == -1: colors.append("#FD9A01")
            elif inf == -2: colors.append("#FD6104")
            else: colors.append("#F00505")
        nx.draw(G[j], with_labels=True, node_color=colors)
    plt.show()

def set_edge_weight(G, i, j):
    x = r()
    if G.nodes[i]["inf"] == 3:
        if x < 0.65: G[i][j]["weight"] = 3
        elif 0.65 <= x < 0.85: G[i][j]["weight"] = 2
        elif 0.85 <= x < 0.95: G[i][j]["weight"] = 1
        elif 0.95 <= x < 0.99: G[i][j]["weight"] = 0
        else: G[i][j]["weight"] = -1
    elif G.nodes[i]["inf"] == 2:
        if x < 0.5: G[i][j]["weight"] = 2
        elif 0.5 <= x < 0.8: G[i][j]["weight"] = 1
        elif 0.8 <= x < 0.95: G[i][j]["weight"] = 0
        else: G[i][j]["weight"] = -1
    elif G.nodes[i]["inf"] == 1:
        if x < 0.5: G[i][j]["weight"] = 1
        elif 0.5 <= x < 0.8: G[i][j]["weight"] = 0
        else: G[i][j]["weight"] = -1
    elif G.nodes[i]["inf"] == 0:
        G[i][j]["weight"] = 0
    elif G.nodes[i]["inf"] == -1:
        if x < 0.5: G[i][j]["weight"] = -1
        elif 0.5 <= x < 0.8: G[i][j]["weight"] = 0
        else: G[i][j]["weight"] = 1
    elif G.nodes[i]["inf"] == -2:
        if x < 0.5: G[i][j]["weight"] = -2
        elif 0.5 <= x < 0.8: G[i][j]["weight"] = -1
        elif 0.8 <= x < 0.95: G[i][j]["weight"] = 0
        else: G[i][j]["weight"] = 1
    elif G.nodes[i]["inf"] == -3:
        if x < 0.65: G[i][j]["weight"] = -3
        elif 0.65 <= x < 0.85: G[i][j]["weight"] = -2
        elif 0.85 <= x < 0.95: G[i][j]["weight"] = -1
        elif 0.95 <= x < 0.99: G[i][j]["weight"] = 0
        else: G[i][j]["weight"] = 1

def prob_inf(G, t, max_iter=500):
    if t not in ["max", "min", "med"]:
        print("Invalid type")
        return
    graphs = [G]
    newNodes = active_nodes(G)
    for i in range(max_iter):
        nxt = prob_inf_step(copy.deepcopy(graphs[-1]), newNodes, t)
        graphs.append(nxt)
        if len(newNodes) == len(G.nodes):
            break
    return graphs

def prob_inf_step(G, new, t):
    step_list = []
    for i in G.nodes:
        if i not in new:
            pred = list(G.predecessors(i))
            for j in pred:
                if j in step_list:
                    return G
                set_edge_weight(G,j,i)
            pw = pred_weights(G, i, pred)
            abs_pw = list(map(abs, pw))
            if sum(abs_pw) == 0:
                continue
            else:
                if t == "max":
                    G.nodes[i]["inf"] = pw[abs_pw.index(max(abs_pw))]
                elif t == "min":
                    G.nodes[i]["inf"] = pw[abs_pw.index(min(abs_pw))]
                elif len(pred) % 2 == 0:
                    G.nodes[i]["inf"] = (pw[len(pw)//2]+pw[len(pw)//2-1])//2
                else:
                    G.nodes[i]["inf"] = pw[len(pw)//2]
            new.append(i)
            step_list.append(i)
    return G
                    
def pred_weights(G, i, pred, nonzero = True):
    ws = []
    for j in pred:
        w = G[j][i]["weight"]
        if not nonzero or w != 0:
            ws.append(w)
    return ws

def active_nodes(G):
    res = []
    for i in G.nodes:
        if G.nodes[i]["inf"] != 0: res.append(i)
    return res

#disinf() and its helper functions do not work for random graphs yet

def disinf(G, t, inf, max_iter=500):
    if t not in ["max", "min", "med"]:
       print("Invalid type")
       return
    graphs = [G]
    newNodes = [15]
    G.nodes[15]["inf"] = inf
    for i in range(max_iter):
        nxt = disinf_step(copy.deepcopy(graphs[-1]), newNodes, t)
        graphs.append(nxt)
        if len(newNodes) == len(G.nodes):
            break
    return graphs, newNodes

def disinf_step(G, new, t):
    step_list = []
    node_list = list(G.nodes)
    node_list.reverse()
    for i in node_list:
        if i not in new:
            succ = list(G.successors(i))
            weights = []
            for j in succ:
                if j in step_list:
                    return G
                weights.append(disinf_weight(G, i, j))
            abs_weights = list(map(abs, weights))    
            if len(weights) != 0:
                if t == "max":
                    G.nodes[i]["inf"] = weights[abs_weights.index(max(abs_weights))]
                elif t == "min":
                    G.nodes[i]["inf"] = weights[abs_weights.index(min(abs_weights))]
                elif len(weights) % 2 == 0:
                    G.nodes[i]["inf"] = (weights[len(weights)//2]+weights[len(weights)//2-1])//2
                else:
                    G.nodes[i]["inf"] = weights[len(weights)//2]
            new.append(i)
            step_list.append(i)
    return G

def disinf_weight(G, i, j):
    x = r()
    if G.nodes[j]["inf"] == G.nodes[i]["inf"]: return G.nodes[j]["inf"]
    elif G.nodes[j]["inf"] + 1 == G.nodes[i]["inf"]:
        if x < 0.5: return G.nodes[i]["inf"]
        else: return G.nodes[j]["inf"]
    elif G.nodes[j]["inf"] + 2 == G.nodes[i]["inf"]:
        if x < 0.5: return G.nodes[i]["inf"]
        elif x < 0.8: return G.nodes[i]["inf"]-1
        else: return G.nodes[j]["inf"]
    elif G.nodes[j]["inf"] + 3 == G.nodes[i]["inf"]:
        if x < 0.5: return G.nodes[i]["inf"]
        elif x < 0.8: return G.nodes[i]["inf"]-1
        elif x < 0.95: return G.nodes[i]["inf"]-2
        else: return G.nodes[j]["inf"]
    else:
        return G.nodes[i]["inf"]
