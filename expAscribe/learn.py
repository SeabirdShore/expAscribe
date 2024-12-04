import pandas as pd
import numpy as np
import copy
from typing import List
from expAscribe.model.voter import Voter
from causallearn.search.FCMBased import lingam
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.PermutationBased.GRaSP import grasp
from scipy import stats
from dagma.linear import DagmaLinear
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression

def preprocess(data:pd.DataFrame):
    data = data.dropna()
    z_scores = stats.zscore(data)
    abs_z_scores = abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data = data[filtered_entries]

    scaler = MinMaxScaler()
    data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data,data_normalized 

def remove_indirect_links(adj_matrix, mi_matrix, threshold):
    n = len(adj_matrix)
    new_adj_matrix = adj_matrix.copy()

    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                for k in range(n):
                    if k != i and k != j and adj_matrix[i][k] == 1 and adj_matrix[k][j] == 1:
                        if mi_matrix[i, j] < mi_matrix[i, k] or mi_matrix[i, j] < mi_matrix[k, j]:
                            new_adj_matrix[i][j] = 0
                            break

    for i in range(n):
        for j in range(n):
            if new_adj_matrix[i][j] <= threshold or i==j:
                new_adj_matrix[i][j] = 0
            else:
                new_adj_matrix[i][j] = 1
    return new_adj_matrix


def pruning(arr:np.ndarray,threshold = 0.1) -> np.ndarray:
    n_features = arr.shape[1]
    mi_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                mi_matrix[i, j] = mutual_info_regression(arr[:, [i]], arr[:, j])
    
    pruned_adj_matrix = remove_indirect_links(arr, mi_matrix, threshold)

    return pruned_adj_matrix

def binarize(arr:np.ndarray,threshold = 0.1) -> np.ndarray:
    cols = arr.shape[1]
    for i in range(cols):
        for j in range(cols):
            if arr[i][j] < 0:
                arr[j][i]-=arr[i][j]
                arr[i][j] = 0
    
    for i in range(cols):
        for j in range(cols):
            if arr[i][j] > threshold:
                arr[i][j] = 1
            else:
                arr[i][j] = 0
    return arr

def voting(candidates:List[np.ndarray], priori:pd.DataFrame) -> np.ndarray:
    # Implement voting algorithm here
    model = Voter(candidates,priori)
    fine_graph = model.run()
    return fine_graph


def ensemble(data:pd.DataFrame) -> List[np.ndarray]:
    # Implement ensemble learning algorithm here
    data,data_normalized = preprocess(data)
    datasrc1 = copy.deepcopy(data)
    datasrc2 = copy.deepcopy(data)
    datasrc3 = copy.deepcopy(data)
    datasrc4 = copy.deepcopy(data)
    candidates = []
    
    lin = lingam.ICALiNGAM(random_state=42,max_iter=2000)
    lin.fit(data.to_numpy())
    glin = lin.adjacency_matrix_
    glin = pruning(glin)
    glin = binarize(glin)
    candidates.append(glin)

    g, edges = fci(datasrc1.to_numpy(),show_progress=False)
    gfci = binarize(g.graph)
    candidates.append(gfci)

    cg = ges(datasrc2.to_numpy())
    gges = binarize(cg['G'].graph)
    candidates.append(gges)

    ggrasp = binarize(grasp(datasrc3.to_numpy()).graph)
    candidates.append(ggrasp)

    model = DagmaLinear(loss_type='l2') 
    gdml = model.fit(datasrc4.to_numpy(), lambda1=0.02)
    gdml = pruning(gdml)
    gdml = binarize(gdml)
    candidates.append(gdml)

    return candidates


