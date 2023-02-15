seed = 0
import random

random.seed(seed)
import numpy as np
import sympy

np.random.seed(seed)
import time
from statistics import mean

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import isomorphism


# generate a random circuit
def generate_circuit(n, num_gates):
    result = []
    # generate labels on every gates
    g = sympy.symbols([f"g{i+1}" for i in range(num_gates)])
    for i in range(num_gates):
        l = random.choices([i for i in range(1, n + 1)], k=2)
        # smaller number in the front
        l.sort()
        # labeling the gates
        l.append(g[i])
        result.append(l)
    return result


# clean up a given circuit
# remove one qubit gates and merge two qubits gates
def clean(random_circuit):
    circuit = []
    for i in range(len(random_circuit)):
        circuit.append(random_circuit[i])
    i = 0
    while i < len(circuit):
        step = circuit[i]
        # makes 1-qubit gate disappear
        if step[0] == step[1]:
            del circuit[i]
        else:
            i = i + 1
    i = 1
    while i < len(circuit):
        # for two identical adjacent 2-qubit gates, makes one disappear
        if (circuit[i][0] - circuit[i - 1][0]) ** 2 + (circuit[i][1] - circuit[i - 1][1]) ** 2 == 0:
            del circuit[i]
        else:
            i = i + 1
    return circuit


# change the order of gates to minimize the number of gates
# see if permutation allows merging more gates
# we do not need this step for minimizing SWAP gates, but it helps reduce the noise
def shuffle_GATE(initial_circuit, s):
    initial_circuit = clean(initial_circuit)
    # L=int(n/2) means try every possible pemutation, can use a smaller number if you want to be faster
    L = int(n / 2)
    circuit = []
    for i in range(len(initial_circuit)):
        circuit.append(initial_circuit[i])
    cost = len(circuit)
    # s is the number for times for shuffle, s=2 is good enough, can take more if you need
    for k in range(s):
        for l in range(2, L + 1):
            # in the case of l=4, if we see (1,2),(3,4),(5,8),(6,7), they do not share a same qubit
            # we can check if the order (6,7),(3,4),(5,8),(1,2) can reduce the number of gates
            # if the gate before is (6,7), it does, and we keep this permutation
            i = 0
            while i < len(circuit) - (l - 1):
                # if l consecutive 2-qubit gates do not share a same qubit, their order can be permuted
                if check_diff(i, circuit, l) == False:
                    new_circuit = permute(i, circuit, l)
                    # merge gates
                    new_circuit = clean(new_circuit)
                    new_cost = len(new_circuit)
                    # check if it needs less gates after exchanged
                    if new_cost < cost:
                        circuit = new_circuit
                        cost = new_cost
                i = i + 1
    return circuit


# generate a random chip for a given number of qubits and edges
def generate_chip(n, num_edges):
    # generate qubit lables for the chip
    Q = sympy.symbols([f"q{i+1}" for i in range(n)])
    if num_edges < n - 1:
        return "Error"
    if num_edges > n * (n - 1) / 2:
        return "Error"
    G = nx.Graph()
    G.add_nodes_from(Q)
    # make sure every qubit is connected
    while nx.is_connected(G) == False:
        G = nx.Graph()
        G.add_nodes_from(Q)
        graph_list = set()
        x = 0
        while x < num_edges:
            List = random.sample([i for i in range(n)], 2)
            List.sort()
            a, b = List
            graph_list.add((Q[a], Q[b]))
            x = len(graph_list)
        G.add_edges_from(graph_list)
    return G


# initialize the circuit using subgraph isomorphism, super slow with more than 12 qubits
# don't pay to much attention on it, not pratical
# let a maximum number of 2-qubit gates can be applied without introducing any SWAP gate
def initial_subgraph(chip, circuit):
    H = nx.Graph()
    n = chip.number_of_nodes()
    H.add_nodes_from([i for i in range(1, n + 1)])
    GM = isomorphism.GraphMatcher(chip, H)
    i = 0
    H.add_edge(circuit[i][0], circuit[i][1])
    # keep adding gates to the first set of gate when it is still a subgraph isomorphism
    # subgraph isomorphism is NP-complete
    while GM.subgraph_is_monomorphic() == True:
        result = GM
        i = i + 1
        H.add_edge(circuit[i][0], circuit[i][1])
        GM = isomorphism.GraphMatcher(chip, H)
        if chip.number_of_edges() == H.number_of_edges() or i == len(circuit) - 1:
            G = nx.relabel_nodes(chip, result.mapping)
            return G, result.mapping
    G = nx.relabel_nodes(chip, result.mapping)
    return G, result.mapping


# initialize the circuit with greedy algorithm
# let a maximum number of 2-qubit gates can be applied without introducing any SWAP gate
def initial_greedy(chip, circuit, g):
    n = chip.number_of_nodes()
    keys = list(chip.nodes())
    values = [i for i in range(1, n + 1)]
    final_mapping = {keys[i]: values[i] for i in range(len(keys))}
    final_G = nx.relabel_nodes(chip, final_mapping)
    final_cost = len(reduce(final_G, circuit))
    for i in range(g):
        random.shuffle(values)
        mapping = {keys[i]: values[i] for i in range(len(keys))}
        G = nx.relabel_nodes(chip, mapping)
        cost = len(reduce(G, circuit))
        if cost == 0:
            return final_G, final_mapping
        if cost < final_cost:
            final_G = G
            final_mapping = mapping
            final_cost = cost
    return final_G, final_mapping


# reduce the circuit
# if a 2-qubit gate can be applied on the current configuration, delete it
def reduce(G, circuit):
    new_circuit = []
    for i in range(len(circuit)):
        new_circuit.append(circuit[i])
    while new_circuit != [] and (new_circuit[0][0], new_circuit[0][1]) in G.edges():
        del new_circuit[0]
    return new_circuit


# for a given path, return all possible walks of qubits
def map_list(path):
    keys = path
    path_ends = [path[0]] + [path[-1]]
    path_middle = path[1:-1]
    List = []
    for i in range(len(path) - 1):
        values = path_middle[:i] + path_ends + path_middle[i:]
        mapping = {keys[i]: values[i] for i in range(len(keys))}
        List.append(mapping)
    return List


# a small greedy algorithm to decide which path to take, and how qubits should walk
def relocate(G, circuit, count_swap):
    if len(circuit) == 0:
        return G, circuit, count_swap
    final_G = G
    n = G.number_of_nodes()
    circuit = reduce(G, circuit)
    final_circuit = circuit
    keys = [i for i in range(1, n + 1)]
    values = keys
    final_mapping = {keys[i]: values[i] for i in range(len(keys))}
    # if a 2-qubit gate could not be placed on the current configuration, find all the shortest path between these 2 qubits
    # the complexity of finding all shortest path on a graph is polynomial
    path_list = [p for p in nx.all_shortest_paths(G, source=circuit[0][0], target=circuit[0][1])]
    count_swap = count_swap + len(path_list[0]) - 2
    # take all shortest paths for greedy (can take less if you want to be faster)
    for i in range(len(path_list)):
        path = path_list[i]
        List = map_list(path)
        for j in range(len(List)):
            mapping = List[j]
            new_G = nx.relabel_nodes(G, mapping)
            new_circuit = reduce(new_G, circuit)
            # greedy looking for the optimal path and the optimal walk on this path
            if len(new_circuit) < len(final_circuit):
                final_G = new_G
                final_circuit = new_circuit
                final_mapping = mapping
    return final_G, final_circuit, final_mapping, count_swap


# give the full transformation of a given chip and a given circuit
# return the number of SWAP gates needed and how to SWAP at each step
def transform(chip, circuit, method, g):
    num_swap = 0
    mapping_list = []
    circuit = clean(circuit)
    if method == "greedy":
        G, mapping = initial_greedy(chip, circuit, g)
    if method == "subgraph":
        G, mapping = initial_subgraph(chip, circuit)
    mapping_list.append(mapping)
    circuit = reduce(G, circuit)
    while len(circuit) != 0:
        G, circuit, mapping, num_swap = relocate(G, circuit, num_swap)
        mapping_list.append(mapping)
    return num_swap, mapping_list


# permute two gates on a circuit
def permute(i, circuit, k):
    new_circuit = []
    for j in range(len(circuit)):
        new_circuit.append(circuit[j])
    new_circuit[i : i + k] = [circuit[i + k - 1]] + circuit[i : i + k - 1]
    return new_circuit


# check if k consecutive 2-qubit gates share a same qubit
def check_diff(i, circuit, k):
    Set = set()
    for j in range(k):
        Set.add(circuit[i + j][0])
        Set.add(circuit[i + j][1])
    if len(Set) == 2 * k:
        return False
    else:
        return True


# change the order of gates to minimize the number of SWAP gates needed
def shuffle_SWAP(s, chip, initial_circuit, method, g):
    n = chip.number_of_nodes()
    # L=int(n/2) means try every possible pemutation, can use a smaller number if you want to be faster
    L = int(n / 2)
    initial_circuit = clean(initial_circuit)
    circuit = []
    for i in range(len(initial_circuit)):
        circuit.append(initial_circuit[i])
    cost, mapping = transform(chip, circuit, method, g)
    # s is the number for times for shuffle, s=2 is good enough, can take more if you need
    for k in range(s):
        for l in range(2, L + 1):
            # in the case of l=4, if we see (1,2),(3,4),(5,8),(6,7), they do not share a same qubit
            # we can check if the order (6,7),(3,4),(5,8),(1,2) can reduce the number of SWAP gates needed
            i = 0
            while i < len(circuit) - (l - 1):
                # if l consecutive 2-qubit gates do not share a same qubit, their order can be permuted
                if check_diff(i, circuit, l) == False:
                    new_circuit = permute(i, circuit, l)
                    # merge gates
                    # new_circuit=clean(new_circuit)
                    new_cost, new_mapping = transform(chip, new_circuit, method, g)
                    # check if it needs less SWAP gates after exchanged
                    if new_cost < cost:
                        circuit = new_circuit
                        cost = new_cost
                        mapping = new_mapping
                i = i + 1
    return cost, circuit, mapping


n = 6

"""
start_time = time.time()


n = 6
num_edges = 6
num_gates = 23

chip = generate_chip(n, num_edges)
plt.figure(1, figsize=(5, 5))
nx.draw(chip, with_labels=True)
plt.savefig("chip.pdf")


#give you and idea of time, this random circuit on our 21-qubit chip takes about one hour to find the solution (1583 SWAPs)
#much longer than a random 21-qubit chip with 32 edges
#probably because the graph is very structured and there is more shortest paths to compare

n=21
num_gates=1000

random.seed(11)
Q=sympy.symbols([f'q{i+1}' for i in range(21)])
chip=nx.Graph()
chip.add_nodes_from(Q)

graph_list_h=[(Q[0],Q[1]),(Q[1],Q[2]),(Q[3],Q[4]),(Q[4],Q[5]),(Q[5],Q[6]),(Q[6],Q[7]),(Q[8],Q[9]),(Q[9],Q[10]),(Q[10],Q[11]),(Q[11],Q[12]),(Q[13],Q[14]),(Q[14],Q[15]),(Q[15],Q[16]),(Q[16],Q[17]),(Q[18],Q[19]),(Q[19],Q[20])]
graph_list_v=[(Q[3],Q[8]),(Q[8],Q[13]),(Q[0],Q[4]),(Q[4],Q[9]),(Q[9],Q[14]),(Q[14],Q[18]),(Q[1],Q[5]),(Q[5],Q[10]),(Q[10],Q[15]),(Q[15],Q[19]),(Q[2],Q[6]),(Q[6],Q[11]),(Q[11],Q[16]),(Q[16],Q[20]),(Q[7],Q[12]),(Q[12],Q[17])]
chip.add_edges_from(graph_list_h+graph_list_v)
plt.figure(1,figsize=(5,5))
pos = nx.spectral_layout(chip)
nx.draw(chip,pos=pos,with_labels = True)
plt.savefig('chip_21.pdf')


EXAMPLE

random_circuit=generate_circuit(n,num_gates)
print('generate random circuit')
print(random_circuit)
print('----------')
initial_circuit=shuffle_GATE(random_circuit,2)
print('minimize the number of gates:', len(initial_circuit))
print(initial_circuit)
print('----------')
cost,circuit,mapping=shuffle_SWAP(2,chip,initial_circuit,'greedy',1000)
print('final circuit, number of gates:', len(circuit))
print(circuit)
print('----------')
print('number of SWAP gates needed:', cost)
print('mapping configuration')
print(mapping)

print("--- %s seconds ---" % (time.time() - start_time))

code for plottting the scaling to compare different methods
you can ignore this part

n=10 #number of qubits
num_edges=20
num_gates=100
num_circuits=10

S=[]
NUM_SWAP_A=[]
NUM_SWAP_B=[]
NUM_SWAP_C=[]
circuit_list=[]
for j in range(num_circuits):
    random_circuit=generate_circuit(n,num_gates)
    circuit_list.append(random_circuit)
for i in range(5):
    data_A=[]
    data_B=[]
    data_C=[]
    for j in range(num_circuits):
        circuit=circuit_list[j]
        circuit=shuffle_GATE(circuit,2)
        #int(n/2) means try every possible pemutation, can use a smaller number
        cost,circuit,mapping=shuffle_SWAP(i,chip,circuit,'subgraph',10)
        data_A.append(cost)
        cost,circuit,mapping=shuffle_SWAP(i,chip,circuit,'greedy',100)
        data_B.append(cost)
        cost,circuit,mapping=shuffle_SWAP(i,chip,circuit,'greedy',1000)
        data_C.append(cost)
    S.append(i)
    NUM_SWAP_A.append(mean(data_A))
    NUM_SWAP_B.append(mean(data_B))
    NUM_SWAP_C.append(mean(data_C))

plt.figure(2,figsize=(8,5))
plt.plot(S,NUM_SWAP_A,'k-',label='subgraph')
plt.plot(S,NUM_SWAP_B,'b-',label='g=100')
plt.plot(S,NUM_SWAP_C,'r-',label='g=1000')

plt.xlabel('num of shuffle')
plt.ylabel('num of SWAP')
plt.legend()
plt.tight_layout()
plt.savefig('Scale.pdf')
"""
