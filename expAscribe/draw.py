import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
#from causalnex.structure import StructureModel
#from causalnex.plots import plot_structure, EDGE_STYLE, NODE_STYLE
from typing import List

def drawgraph(adj_matrix:np.ndarray, labels:List[str]):
    labels_dict = {i: label for i, label in enumerate(labels)}

    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    pos = nx.kamada_kawai_layout(G)  

    nx.draw(G, pos, with_labels=True, labels=labels_dict, node_color='lightblue', arrowsize=20, edge_color='gray')
    nx.draw_networkx_labels(G, pos, labels=labels_dict)
    plt.show()
    '''
    if len(node_labels) != adj_matrix.shape[0]:
        raise ValueError("dimensions not match")

    sm = StructureModel()

    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1:
                sm.add_edge(node_labels[i], node_labels[j])

    g = plot_structure(
        sm,
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK,
    )
    g.show("graph.html")
    with open("graph.html", "r") as f:
        return f.read()
    '''


def multiline(results:List[pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for df in results:
        x = df[df.columns[0]]  
        y = df[df.columns[1]] 
        lower_ci = df['Lower_CI']
        upper_ci = df['Upper_CI']
        
        ax.plot(x, y, label=f'{df.columns[1]}', marker='o')
        
        ax.fill_between(x, lower_ci, upper_ci, alpha=0.2)
    
    ax.set_xlabel('Intervention')
    ax.set_ylabel('Causal Dose Response')
    ax.set_title('Causal Dose Response with Confidence Intervals')
    ax.legend()
    
    plt.show()


