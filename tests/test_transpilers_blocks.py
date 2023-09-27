from qibo import Circuit, gates

from qibolab.transpilers.blocks import initial_block_deomposition

circ = Circuit(4)
circ.add(gates.H(0))
circ.add(gates.H(1))
circ.add(gates.CZ(0, 1))
circ.add(gates.H(0))
# circ.add(gates.H(1))
# circ.add(gates.CZ(2,3))
# circ.add(gates.H(2))
# circ.add(gates.CZ(2,1))
# circ.add(gates.H(1))
# #circ.add(gates.CZ(2,1))
# circ.add(gates.CZ(2,0))

# dag = create_dag(circ)
# pos = nx.multipartite_layout(dag, subset_key="layer")
# fig, ax = plt.subplots()
# nx.draw_networkx(dag, pos=pos, ax=ax)
# ax.set_title("DAG layout in topological order")
# fig.tight_layout()
# plt.show()

initial_block_deomposition(circ)
