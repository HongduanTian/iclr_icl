import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

def svm(data:np.ndarray, labels:np.array, queries:np.ndarray, seed:int) -> np.ndarray:
    """
    SVM algorithm.

    Args:
        data (np.ndarray): Training data.
        labels (np.array): Labels of the training data.
        queries (np.ndarray): Test data.
        seed (int): Random seed.

    Returns:
        np.ndarray: Prediction of the test data.
    """
    num_classes = np.unique(labels).shape[0]
    assert num_classes > 1, f"The number of classes should be larger than 1."
    
    if num_classes == 2:
        dfs = "ovr"
    else:
        dfs = "ovo"
    
    ### TODO: gamma 0.001 - 7.0 have interesting phenomenon
    rbf_svm = SVC(kernel="rbf", gamma=0.5, decision_function_shape=dfs, random_state=seed)    # "ovr" means "one-vs-rest"
    rbf_svm.fit(data, labels)
    pred = rbf_svm.predict(queries)
    
    return pred


def knn(data:np.ndarray, labels:np.array, queries:np.ndarray, seed:int) -> np.ndarray:
    """
    KNN classification algorithm.

    Args:
        data (np.ndarray): Training data.
        labels (np.array): Labels of the training data.
        queries (np.ndarray): Test data.

    Returns:
        np.ndarray: Prediction of the test data.
    """
    num_classes = np.unique(labels).shape[0]
    assert num_classes > 1, f"The number of classes should be larger than 1."
    
    knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto").fit(data, labels)
    pred = knn.predict(queries)
    return pred

def decisiontree(data:np.ndarray, labels:np.array, queries:np.ndarray, seed:int) -> np.ndarray:
    """
    Decision tree algorithm.

    Args:
        data (np.ndarray): Training data.
        labels (np.array): Labels of the training data.
        queries (np.ndarray): Test data.
        seed (int): Random seed.

    Returns:
        np.ndarray: Prediction of the test data.
    """
    num_classes = np.unique(labels).shape[0]
    assert num_classes > 1, f"The number of classes should be larger than 1."
    
    decision_tree = tree.DecisionTreeClassifier(max_depth=10, random_state=seed).fit(data, labels)
    pred = decision_tree.predict(queries)
    return pred


def mlp(data:np.ndarray, labels:np.array, queries:np.ndarray, seed:int) -> np.ndarray:
    """
    MLP classification model.

    Args:
        data (np.ndarray): Training data.
        labels (np.array): Labels of the training data.
        queries (np.ndarray): Test data.
        seed (int): Random seed for weights initialization.

    Returns:
        np.ndarray: Prediction of the test data.
    """
    num_classes = np.unique(labels).shape[0]
    assert num_classes > 1, f"The number of classes should be larger than 1."
    
    mlp = MLPClassifier(hidden_layer_sizes=(256,), random_state=seed, max_iter=1000).fit(data, labels)
    pred = mlp.predict(queries)
    return pred


def linear_regression(data:np.ndarray, labels:np.ndarray, queries:np.ndarray, seed:int) -> np.ndarray:
    """
    Linear regression method.

    Args:
        data (np.ndarray): Training data.
        labels (np.ndarray): Labels of the training data.
        queries (np.ndarray): Test data.

    Returns:
        np.ndarray: predictions of queries.
    """
    reg = LinearRegression().fit(data, labels)
    pred = reg.predict(queries)
    return pred