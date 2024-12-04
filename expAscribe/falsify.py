import pandas as pd
import networkx as nx
from sklearn.ensemble import GradientBoostingRegressor
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.independence_test.generalised_cov_measure import generalised_cov_based
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.ml import SklearnRegressionModel

def create_gradient_boost_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(GradientBoostingRegressor(**kwargs))

def gcm(X, Y, Z=None):
    return generalised_cov_based(X, Y, Z=Z, prediction_model_X=create_gradient_boost_regressor,
                                        prediction_model_Y=create_gradient_boost_regressor)
def falsifypack(
        data_path, graph_path, to_plot=True,
        significance_level_=0.05,
        significance_ci_=0.05,
        n_perm=100) -> str:
    set_random_seed(1332)
    data = pd.read_csv(data_path)
    g = nx.read_gml(graph_path)
    result = falsify_graph( g, data, 
                            n_permutations=n_perm,
                            independence_test=gcm,
                            conditional_independence_test=gcm,
                            significance_level=significance_level_,
                            significance_ci=significance_ci_,
                            plot_histogram=to_plot)

    return result