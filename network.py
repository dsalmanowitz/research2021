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

def show(G, r = False):
    if type(G) != list: G = [G]
    if r: G.reverse()
    for j in range(len(G)):
        plot = plt.figure(len(G)-j if r else j+1)
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
            G.nodes[i]["inf"] += 5*s
            G.nodes[i]["inf"] /= 6
    return G, total_inf(G)

def lin_thresh(G):
    graphs = [G]
    infs = [total_inf(G)]
    while True:
        nxt, inf = lin_thresh_step(copy.deepcopy(graphs[-1]))
        if abs(inf - infs[-1]) < 0.0001 or inf in infs:
            break
        graphs.append(nxt)
        infs.append(inf)
    return graphs, infs

# Metrics

def num_inf(G):
    n = 0
    p = 0
    for i in G.nodes:
        inf = G.nodes[i]["inf"]
        if inf < 0: n += 1
        elif inf > 0: p += 1
    return [n,p]

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

#Degree Centrality
def cd(G,n):
    return G.degree(n) / (len(G.nodes) - 1)

def most_cd(G):
    high = 0
    high_c = 0
    for i in G.nodes:
        c = cd(G,i)
        if c > high_c:
            high = i
            high_c = c
    return high, high_c

#Closeness Centrality
def cc(G,n):
    total = 0
    for i in G.nodes:
        if nx.has_path(G, n, i):
            total += nx.shortest_path_length(G, n, i)
    return round(1 / total, 2) if total != 0 else 0

def most_cc(G):
    high = 0
    high_c = 0
    for i in G.nodes:
        c = cc(G,i)
        if c > high_c:
            high = i
            high_c = c
    return high, high_c   

#gs, infs = lin_thresh(get_graph(n=20,p=0.2))
#print("Summed influence: " + str(total_inf(gs[-1])))
#print(str(len(infs)) + " iterations")
#show(gs[-1])

