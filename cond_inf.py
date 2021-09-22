import matplotlib.pyplot as plt
import networkx as nx
import random
import copy
import numpy as np
import pandas as pd
import math
from collections import Counter

def r():
    return random.random()

#Returns a simple directed graph. Used for testing influence spread.
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

#Returns a directed graph using n nodes with binomial probability for edges p. Used for testing two-sided influence spread.
def get_graph1(n=16, p=0.3):
    G = nx.fast_gnp_random_graph(n, p, directed=True)
    for i in G.nodes:
        G.nodes[i]["inf"] = 0
    G.nodes[0]["inf"] = 3
    G.nodes[1]["inf"] = -3
    for (start, end) in G.edges:
        G.edges[start, end]["weight"] = 0
    return G

#Creates a graph of the London power network using data from Raman et al
#Input: area subsection and node limiter (default including all nodes)
def create_london(area=0, n=0):
    if area < 0 or area > 8:
        print("Invalid area of Greater London")
        return
    xls = pd.ExcelFile("GreaterLondon_Power_Network.xlsx")
    labels = ["From", "To", "Buildings", "Type", "Length"]
    gl = pd.read_excel(xls, "Greater London " + str(area), names=labels)
    subset = gl.drop(axis=0, index=0) if n < 1 else gl[(gl.From < n) & (gl.To < n)]
    edges = subset[["From","To"]].values.tolist()
    return nx.from_edgelist(edges)

#Outputs a network graph visualization for one-sided influence spread.
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
    
#Outputs a network graph visualization for two-sided influence spread. 
def show1(G):
    if type(G) != list:
        G = [G]
    for j in range(len(G)):
        plot = plt.figure(j+1)
        colors = []
        for i in G[j].nodes:
            inf = G[j].nodes[i]["inf"]
            if inf == 3: colors.append("#4D8C57")
            elif inf == 2: colors.append("#04FDFD")
            elif inf == 1: colors.append("#0429FD")
            elif inf == 0: colors.append("#FFFF00")
            elif inf == -1: colors.append("#FD04F9")
            elif inf == -2: colors.append("#FD6104")
            else: colors.append("#F00505")
        nx.draw(G[j], with_labels=True, node_color=colors)
    plt.show()
    
#Used in prob_inf(): sets the weight of an edge to a connected node j given the influence level of one node i
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

def max_inf(G):
    return prob_inf(G, "max")
def min_inf(G):
    return prob_inf(G, "min")
def med_inf(G):
    return prob_inf(G, "med")

#Influences a graph using maximum, minimum, or median influence propogation.
#Returns a list of graphs for every step in the algorithm.
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

#Influences the graph for one step.
#Returns the new graph.
def prob_inf_step(G, new, t):
    step_list = []
    for i in G.nodes:
        if i not in new:
            #pred = list(G.predecessors(i)) #for directed graphs
            pred = list(G.neighbors(i)) #for undirected London graph
            for j in pred:
                if j > i:
                    continue
                if j in step_list:
                    return G
                set_edge_weight(G,j,i)
            pw = pred_weights(G, i, pred)
            abs_pw = list(map(abs, pw))
            if sum(abs_pw) == 0:
                continue
            else: #given the weights of connected edges and an inputted propogation method, assign an influence to the next node
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

#Used in prob_inf(): Returns the weights of all connected edges from predecessor nodes     
def pred_weights(G, i, pred, nonzero = True):
    ws = []
    for j in pred:
        w = G[j][i]["weight"]
        if not nonzero or w != 0:
            ws.append(w)
    return ws

#Used in prob_inf(): Returns a list of the nodes that are currently influenced
def active_nodes(G):
    res = []
    for i in G.nodes:
        if G.nodes[i]["inf"] != 0: res.append(i)
    return res

#Disinfluences a graph using maximum, minimum, or median influence propogation.
#Returns a list of graphs for every step in the algorithm.
def disinf(G, t, p, inf, max_iter=500):
    if t not in ["max", "min", "med"]:
       print("Invalid type")
       return
    graphs = [G]
    newNodes = disinf_start_nodes(G, p)
    for i in newNodes:
        G.nodes[i]["inf"] = inf
    for i in range(max_iter):
        nxt = disinf_step(copy.deepcopy(graphs[-1]), newNodes, t)
        graphs.append(nxt)
        if len(newNodes) == len(G.nodes):
            break
    return graphs

#Returns a list of chosen inner nodes in graph G to be disinfluenced, given 0<=p<=1 * len(G) inner nodes should be disinfluenced
def disinf_start_nodes(G, p):
    eligible_nodes = list(filter(lambda i: G.out_degree(i) != 0, G.nodes))
    max_size = len(eligible_nodes)*p//1
    count = 0
    chosen_nodes = []
    while count < max_size:
        potential_node = r()*len(eligible_nodes)//1
        if potential_node not in chosen_nodes:
            chosen_nodes.append(potential_node)
            count+=1
    return chosen_nodes

#Disinfluences the graph for one step.
#Returns the new graph.
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

#Returns the weight of an edge to a connected node i given the influence level of one node j
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

#Returns a graph reflecting the average influence value of each node given a list of graphs with the same structure
def average_inf(graphs):
    node_res = [0 for i in graphs[0].nodes]
    for G in graphs:
        for n in G.nodes:
            node_res[n] += G.nodes[n]["inf"]
    node_res = np.array(node_res)/len(graphs)
    avg_graph = copy.deepcopy(graphs[0])
    for n in avg_graph.nodes:
        avg_graph.nodes[n]["inf"] = round(node_res[n])
    return avg_graph

#Simulates the usage of a battery for an electric car at each 'house' in a network.
#Returns the list of overloaded nodes.
#TODO: If a node is overloaded, its successors do not receive any power either.
def energy_usage(G):

    P_rated = 3.3   #value in kW
    E_rated = 7        #value in kWH
    Bat_Efficiency = .85 
    E_bat = .5 * E_rated #initial amount of energy stored in battery on 1st day
    delta_t = 1/15
    P_bat = 0
    P_bat_list = []       #empty list for graphing energy values later
    Tx_power = pd.read_csv("Single Day October 31.csv")
        
    H = Tx_power["dataid"]                  #unique id for household data
    d = Tx_power["day number"]              #what day of the year
    m = Tx_power["time quarter number"]     #what minute of the day 

    dead_nodes = []
        
    t = 0  #for loop counter

    for i in G.nodes:
        t = 0
        P_bat = 0
        P_bat_list = []
        E_bat = .5 * E_rated
        chargecount = 0
        for j in Tx_power["time quarter number"]:
            P_use = Tx_power["grid1"]         #energy used calculated in excel as sum of all eguage values except grid and solar
            P_gen = Tx_power["solar1"]
            P_target = P_use[t]-P_gen[t]
            
            if P_target > 0 and E_bat >= 0:
               P_bat = P_target
            elif P_target < 0 and E_bat <= E_rated:
                P_bat = max(P_target, -P_rated)
            else : 
                P_bat = 0

            #The battery statistics run as normal, but the only factor used in deciding whether to charge or discharge the battery is the level of influence
            #(the higher the influence, the more likely the battery is charged)

            x=r()
            if  (G.nodes[i]["inf"] == 3 and x < 0.2) or (G.nodes[i]["inf"] == 2 and x < 0.4) or (G.nodes[i]["inf"] == 1 and x < 0.7) or (G.nodes[i]["inf"] == 0):
                k = 1/math.sqrt(Bat_Efficiency)     #discharging
            else:
                chargecount = chargecount+1;
                k = math.sqrt(Bat_Efficiency)       #charging
                
            E_bat = E_bat - P_bat * k * delta_t
            t = t + 1

            if chargecount > 1.2*61: #The battery charges 61 times in a normal day. If it charges (example: 20%) more often than normal, it is overloaded.
                list.append(dead_nodes, i)
                break

    return dead_nodes

#Runs 100 tests in which the London subnetwork is influenced and electricity output is measured
#Returns a list of the most commonly overloaded nodes and the number of times they overloaded
def energy_test_run():
    c = create_london(0, 50)
    c.add_edges_from([(8,46), (13, 38), (11, 1), (3, 43), (37, 23), (37, 48), (23, 33), (15, 30)]) #to connect all nodes
    for i in c.nodes:
        c.nodes[i]["inf"] = 0
    c.nodes[7]["inf"] = 3
    for (start, end) in c.edges:
        c.edges[start, end]["weight"] = 0
    all_dead_nodes = []
    for i in list(range(100)):
        c = max_inf(c)[-1]
        dead_nodes = energy_usage(c)
        list.append(all_dead_nodes, dead_nodes)
    counter = Counter(all_dead_nodes[0])
    for i in all_dead_nodes[1:]:
        counter.update(i);

    return counter.most_common()


#Metrics

#Positive Only
def num_inf(G):
    res = [0 for i in range(4)]
    for i in G.nodes:
        inf = G.nodes[i]["inf"]
        if inf >= 0: res[inf] += 1
    return res

def total_inf(G):
    s = 0
    for i in G.nodes:
        s += G.nodes[i]["inf"]
    return s

def mean_g_inf(G):
    return total_inf(G) / len(list(G.nodes))

def highest_metric(G, f):
    high = 0
    high_val = 0
    for i in G.nodes:
        n = f(G,i)
        if n > high_val:
            high = i
            high_val = n
    return high, high_val

#Outdegree
def out(G,i):
    return len(list(G.successors(i)))

def most_out(G):
    return highest_metric(G, out)

#Degree Centrality
def cd(G,i):
    return G.degree(i) / (len(G.nodes) - 1)

def most_cd(G):
    return highest_metric(G, cd)

#Closeness Centrality
def cc(G,i):
    total = 0
    for j in G.nodes:
        if nx.has_path(G, i, j):
            total += nx.shortest_path_length(G, i, j)
    return round(1 / total, 2) if total != 0 else 0

def most_cc(G):
    return highest_metric(G, cc)
