import matplotlib.pyplot as plt
import networkx as nx
import random
import copy

def r():
    return random.random()

def get_graph(n=5, p=0.5):
    G = nx.fast_gnp_random_graph(n, p, directed=True)
    for i in G.nodes:
        G.nodes[i]["inf"] = "n"
        G.nodes[i]["thresh"] = r()
    G.nodes[most_suc(G)[0]]["inf"] = "y"
    for (start, end) in G.edges:
        G.edges[start, end]["weight"] = r()
    return G

def show(G):
    colors = []
    #for i in G.nodes:
    #    inf = G.nodes[i]["inf"]
    #    if inf < 0.2: colors.append("#ED2938")
    #    elif inf < 0.4: colors.append("#B25F4A")
    #    elif inf < 0.6: colors.append("#77945C")
    #    elif inf < 0.8: colors.append("#3BCA6D")
    #    else: colors.append("#00FF7F")
    for i in G.nodes:
        colors.append("lightcoral" if G.nodes[i]["inf"] == "n" else "yellowgreen")
    nx.draw(G, with_labels=True, node_color=colors)
    plt.show()

# Cascade

def cascade_step(G, active):
    new = []
    for i in active:
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

# Linear Threshold

def lin_thresh_step(G):
    new = []
    for i in G.nodes:
        if G.nodes[i]["inf"] == "n":
            pred = list(G.predecessors(i))
            sum = 0
            for j in pred:
                sum += G[j][i]["weight"]
            if sum/len(pred) > G.nodes[i]["thresh"]:
                G.nodes[i]["inf"] = "y"
                new.append(i)
    return G, new

def lin_thresh(G):
    graphs = [G]
    newNodes = [[most_suc(G)[0]]]
    while True:
        nxt, new = lin_thresh_step(copy.deepcopy(graphs[-1]))
        if new == []:
            break
        graphs.append(nxt)
        newNodes.append(new)
    return graphs, newNodes

# Metrics

def num_influenced(G):
    count = 0
    for i in G.nodes:
        if G.nodes[i]["inf"] > 0.5:
            count += 1
    return count

def perc_influenced(G):
    return num_influenced(G)/len(list(G.nodes))

def most_suc(G):
    most = 0
    most_suc = 0
    for i in G.nodes:
        n = len(list(G.successors(i)))
        if n > most_suc:
            most = i
            most_suc = n
    return most, most_suc
