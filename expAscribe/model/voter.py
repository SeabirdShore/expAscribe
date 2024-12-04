import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter,deque
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, average_precision_score, recall_score, f1_score, accuracy_score
class Voter:
    def __init__(self, adj_matrices, y_meta):
        self.adj_matrices = adj_matrices
        self.y_meta = y_meta
        self.process_y_meta()  
        self.num_nodes = adj_matrices[0].shape[0]
        self.feature_matrix = self.create_feature_matrix()
        self.target_vector = self.y_meta.values.flatten()
        self.meta_model = LogisticRegression()

    def create_feature_matrix(self):

        num_matrices = len(self.adj_matrices)
        num_elements = self.num_nodes * self.num_nodes
        X = np.zeros((num_elements, num_matrices))
        for i, matrix in enumerate(self.adj_matrices):
            X[:, i] = matrix.flatten()
        return X

    def process_y_meta(self):

        y_meta_np = self.y_meta.values
        num_nodes = y_meta_np.shape[0]
        for start in range(num_nodes):
            queue = deque([start])
            visited = set()
            while queue:
                current = queue.popleft()
                for neighbor in range(num_nodes):
                    if y_meta_np[current, neighbor] == 1 and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        y_meta_np[start, neighbor] = 1

        self.y_meta = pd.DataFrame(y_meta_np, index=self.y_meta.index, columns=self.y_meta.columns)

    def find_best_threshold(self, cv_folds=5):
        def evaluate_threshold(threshold):
            predictions = (self.meta_model.predict_proba(self.feature_matrix)[:, 1] >= threshold).astype(int)
            return accuracy_score(self.target_vector, predictions)

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        best_threshold = 0
        best_score = 0

        for threshold in np.linspace(0, 1, 100):
            scores = []
            for train_idx, val_idx in kf.split(self.feature_matrix):
                X_train_cv, X_val_cv = self.feature_matrix[train_idx], self.feature_matrix[val_idx]
                y_train_cv, y_val_cv = self.target_vector[train_idx], self.target_vector[val_idx]
                self.meta_model.fit(X_train_cv, y_train_cv)
                score = evaluate_threshold(threshold)
                scores.append(score)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = threshold

        return best_threshold

    def run(self):
        #X_train, X_test, y_train, y_test = train_test_split(self.feature_matrix, self.target_vector, test_size=0.2, random_state=42)
        self.meta_model.fit(self.feature_matrix, self.target_vector)
        weights = self.meta_model.coef_.flatten()

        best_threshold = self.find_best_threshold()
        
        final_adj_matrix = self.generate_final_adj_matrix(weights, best_threshold)

        return final_adj_matrix

    def generate_final_adj_matrix(self, weights, threshold):
        num_elements = self.num_nodes * self.num_nodes
        weighted_sum = np.zeros(num_elements)
        for i, matrix in enumerate(self.adj_matrices):
            weighted_sum += weights[i] * matrix.flatten()

        final_adj_matrix_flat = (weighted_sum >= threshold).astype(int)
        final_adj_matrix = final_adj_matrix_flat.reshape((self.num_nodes, self.num_nodes))

        return final_adj_matrix
