import matplotlib.pyplot as plt
import networkx as nx
import random
import copy

def r():
    return random.random()

def get_graph(n=5, p=0.5):
    G = nx.fast_gnp_random_graph(n, p, directed=True)
    for i in G.nodes:
        G.nodes[i]["inf"] = 0
        G.nodes[i]["thresh"] = r()
    G.nodes[0]["inf"] = 1
    G.nodes[1]["inf"] = -1
    for (start, end) in G.edges:
        G.edges[start, end]["weight"] = r()
    return G

def show(G):
    if type(G) != list:
        G = [G]
    for j in range(len(G)):
        plot = plt.figure(j+1)
        colors = []
        for i in G[j].nodes:
            inf = G[j].nodes[i]["inf"]
            if inf < -0.5: colors.append("#FF0000")
            elif inf < 0: colors.append("#FF8000")
            elif inf == 0: colors.append("#FFFF00")
            elif inf < 0.5: colors.append("#80FFAA")
            else: colors.append("#00FF00")     
        nx.draw(G[j], with_labels=True, node_color=colors)
    plt.show()

# Cascade

#Does not yet work with disinfection
def cascade_step(G, active):
    new = []
    for i in active:
        suc = list(G.successors(i))
        for j in suc:
            if G.nodes[j]["inf"] > 0 and G[i][j]["weight"] > G.nodes[j]["thresh"]:
                G.nodes[j]["inf"] += G[i][j]["weight"] - G.nodes[j]["thresh"]
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

# Linear Threshold

def lin_thresh_step(G):
    new = []
    for i in G.nodes:
        pred = list(G.predecessors(i))
        s = 0
        p = 0
        for j in pred:
            s += G[j][i]["weight"] * G.nodes[j]["inf"]
            if G.nodes[j]["inf"] != 0:
                p += 1
        s /= p if p != 0 else 1
        if abs(s) > G.nodes[i]["thresh"]:
            G.nodes[i]["inf"] += s
            G.nodes[i]["inf"] /= 2
            new.append(i)
    return G, new

def lin_thresh(G):
    graphs = [G]
    newNodes = [[0, 1]]
    while True:
        nxt, new = lin_thresh_step(copy.deepcopy(graphs[-1]))
        if new == [] or new in newNodes:
            break
        graphs.append(nxt)
        newNodes.append(new)
    return graphs, newNodes

# Metrics

def total_inf(G):
    s = 0
    for i in G.nodes:
        s += G.nodes[i]["inf"]
    return s

def avg_inf(G):
    return total_inf(G) / len(list(G.nodes))

def most_suc(G):
    most = 0
    most_suc = 0
    for i in G.nodes:
        n = len(list(G.successors(i)))
        if n > most_suc:
            most = i
            most_suc = n
    return most, most_suc

gs, ns = lin_thresh(get_graph(n=20))
print(total_inf(gs[-1]))
