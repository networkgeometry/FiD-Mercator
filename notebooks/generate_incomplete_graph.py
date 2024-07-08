import networkx as nx
import numpy as np


def is_giant_connected_component(g):
    gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g0 = g.subgraph(gcc[0])
    return nx.number_of_nodes(g0) == nx.number_of_nodes(g)
    
    
def generate_incomplete_network(g, q0=0.1, check_gcc=True):
    # q0 -- fraction of missing links from original graph
    E_total = nx.number_of_edges(g)
    E_remove = int(E_total * q0)
    
    all_edges = list(g.edges())
    idx = np.arange(0, len(all_edges), dtype=int)

    g_new = g.copy()
    n_edges_removed = 0
    random_edges = np.random.choice(idx, size=E_total, replace=False)
    
    for i in random_edges:
        e = all_edges[i]
        if n_edges_removed == E_remove:
            break
        g_new.remove_edge(*e)
        if check_gcc:
            if not is_giant_connected_component(g_new):
                g_new.add_edge(*e)
            else:
                n_edges_removed += 1
        else: # might lead to isolated nodes
            n_edges_removed += 1

    return g_new


def generate_incomplete_network_gcc(g, q=0.1):
    # q0 -- fraction of missing links from original graph
    E_total = nx.number_of_edges(g)
    E_remove = int(E_total * q)

    all_edges = list(g.edges())
    idx = np.arange(0, len(all_edges), dtype=int)

    g_new = g.copy()
    random_edges = np.random.choice(idx, size=E_remove, replace=False)

    missing_links = []
    for i in random_edges:
        e = all_edges[i]
        missing_links.append(e)
        g_new.remove_edge(*e)

    gcc = sorted(nx.connected_components(g_new), key=len, reverse=True)
    g0 = g_new.subgraph(gcc[0])
    print(f"Before: {nx.number_of_nodes(g)} nodes. After: {nx.number_of_nodes(g0)} nodes")
    return g0, missing_links


def generate_bunch_incomplate_graph(g, q0=0.1, ntimes=5):
    return [generate_incomplete_network(g, q0=q0) for _ in range(ntimes)]