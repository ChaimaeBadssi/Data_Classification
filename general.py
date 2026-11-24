#all functions used
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def best_k(X_train, y_train, X_val, y_val, max_k=20):
    """
    Find the best value of k for KNN classifier using cross-validation.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_val: Validation features
    - y_val: Validation labels
    - max_k: Maximum value of k to test

    Returns:
    - best_k: The value of k that gives the highest accuracy
    - best_accuracy: The highest accuracy achieved
    """
    best_accuracy = 0
    best_k = 1

    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_val, y_val)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return best_k, best_accuracy


def knn_model(X_train,y_train,X_test,y_test,k):
    """
    Train and evaluate a KNN classifier.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Testing features
    - y_test: Testing labels
    - k: Number of neighbors for KNN

    Returns:
    - accuracy: Accuracy of the model on the test set
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    
    return knn, accuracy


def logistic_regression_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Logistic Regression classifier.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Testing features
    - y_test: Testing labels

    Returns:
    - accuracy: Accuracy of the model on the test set
    """
    from sklearn.linear_model import LogisticRegression

    lgs = LogisticRegression(max_iter=1000)
    lgs.fit(X_train, y_train)
    accuracy = lgs.score(X_test, y_test)

    return lgs, accuracy


def best_svm_kernel_C(X_train, y_train, X_val, y_val, kernels=['linear', 'poly', 'rbf', 'sigmoid'], C_values=[0.1, 1, 10]):
    """
    Find the best kernel and C value for SVM classifier using cross-validation.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_val: Validation features
    - y_val: Validation labels
    - kernels: List of kernels to test
    - C_values: List of C values to test

    Returns:
    - best_kernel: The kernel that gives the highest accuracy
    - best_C: The C value that gives the highest accuracy
    - best_accuracy: The highest accuracy achieved
    """
    from sklearn.svm import SVC

    best_accuracy = 0
    best_kernel = None
    best_C = None

    for kernel in kernels:
        for C in C_values:
            svm = SVC(kernel=kernel, C=C)
            svm.fit(X_train, y_train)
            accuracy = svm.score(X_val, y_val)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_kernel = kernel
                best_C = C

    return best_kernel, best_C, best_accuracy


def svm_model(X_train, y_train, X_test, y_test, best_kernel, best_C):
    """
    Train and evaluate a Support Vector Machine (SVM) classifier.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Testing features
    - y_test: Testing labels
    - best_kernel: The kernel to use for the SVM
    - best_C: The C value to use for the SVM

    Returns:
    - accuracy: Accuracy of the model on the test set
    """
    from sklearn.svm import SVC

    svm = SVC(kernel=best_kernel, C=best_C)
    svm.fit(X_train, y_train)
    accuracy = svm.score(X_test, y_test)

    return svm, accuracy


def best_random_forest_hyperparameters(X_train, y_train, X_val, y_val, n_estimators_list=[10, 50, 100], max_depth_list=[None, 5, 10]):
    """
    Find the best hyperparameters for Random Forest classifier using cross-validation.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_val: Validation features
    - y_val: Validation labels
    - n_estimators_list: List of n_estimators to test
    - max_depth_list: List of max_depth values to test

    Returns:
    - best_n_estimators: The n_estimators that gives the highest accuracy
    - best_max_depth: The max_depth that gives the highest accuracy
    - best_accuracy: The highest accuracy achieved
    """
    from sklearn.ensemble import RandomForestClassifier

    best_accuracy = 0
    best_n_estimators = None
    best_max_depth = None

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            rf.fit(X_train, y_train)
            accuracy = rf.score(X_val, y_val)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_n_estimators = n_estimators
                best_max_depth = max_depth

    return best_n_estimators, best_max_depth, best_accuracy


def random_forest_model(X_train, y_train, X_test, y_test, best_n_estimators, best_max_depth):
    """
    Train and evaluate a Random Forest classifier.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Testing features
    - y_test: Testing labels
    - best_n_estimators: The n_estimators to use for the Random Forest
    - best_max_depth: The max_depth to use for the Random Forest

    Returns:
    - accuracy: Accuracy of the model on the test set
    """
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth)
    rf.fit(X_train, y_train)
    accuracy = rf.score(X_test, y_test)

    return rf, accuracy


def best_decision_tree_hyperparameters(X_train, y_train, X_val, y_val, max_depth_list=[None, 5, 10], min_samples_split_list=[2, 5, 10]):
    """
    Find the best hyperparameters for Decision Tree classifier using cross-validation.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_val: Validation features
    - y_val: Validation labels
    - max_depth_list: List of max_depth values to test
    - min_samples_split_list: List of min_samples_split values to test

    Returns:
    - best_max_depth: The max_depth that gives the highest accuracy
    - best_min_samples_split: The min_samples_split that gives the highest accuracy
    - best_accuracy: The highest accuracy achieved
    """
    from sklearn.tree import DecisionTreeClassifier

    best_accuracy = 0
    best_max_depth = None
    best_min_samples_split = None

    for max_depth in max_depth_list:
        for min_samples_split in min_samples_split_list:
            dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
            dt.fit(X_train, y_train)
            accuracy = dt.score(X_val, y_val)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_max_depth = max_depth
                best_min_samples_split = min_samples_split

    return best_max_depth, best_min_samples_split, best_accuracy


def decision_tree_model(X_train, y_train, X_test, y_test, best_max_depth, best_min_samples_split):
    """
    Train and evaluate a Decision Tree classifier.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Testing features
    - y_test: Testing labels
    - best_max_depth: The max_depth to use for the Decision Tree
    - best_min_samples_split: The min_samples_split to use for the Decision Tree

    Returns:
    - accuracy: Accuracy of the model on the test set
    """
    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_split=best_min_samples_split)
    dt.fit(X_train, y_train)
    accuracy = dt.score(X_test, y_test)

    return dt, accuracy


from sklearn.ensemble import GradientBoostingClassifier

def best_gradient_boosting_hyperparameters(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators_list=[50, 100, 200],
    learning_rate_list=[0.01, 0.1, 0.2],
    max_depth_list=[1, 3, 5]
):
    """
    Find the best hyperparameters for Gradient Boosting.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_val: Validation features
    - y_val: Validation labels
    - n_estimators_list: Number of trees to test
    - learning_rate_list: Learning rates to test
    - max_depth_list: Tree depths to test

    Returns:
    - best_n_estimators
    - best_learning_rate
    - best_max_depth
    - best_accuracy
    """

    best_accuracy = 0
    best_n_estimators = None
    best_learning_rate = None
    best_max_depth = None

    for n_estimators in n_estimators_list:
        for learning_rate in learning_rate_list:
            for max_depth in max_depth_list:

                gb = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42
                )

                gb.fit(X_train, y_train)

                accuracy = gb.score(X_val, y_val)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_n_estimators = n_estimators
                    best_learning_rate = learning_rate
                    best_max_depth = max_depth

    return (
        best_n_estimators,
        best_learning_rate,
        best_max_depth,
        best_accuracy
    )

def gradient_boosting_model(
    X_train,
    y_train,
    X_test,
    y_test,
    best_n_estimators,
    best_learning_rate,
    best_max_depth
):
    """
    Train and evaluate a Gradient Boosting classifier.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Testing features
    - y_test: Testing labels
    - best_n_estimators: Number of boosting stages
    - best_learning_rate: Learning rate
    - best_max_depth: Maximum depth of trees

    Returns:
    - gb: Trained model
    - accuracy: Test accuracy
    """

    from sklearn.ensemble import GradientBoostingClassifier

    gb = GradientBoostingClassifier(
        n_estimators=best_n_estimators,
        learning_rate=best_learning_rate,
        max_depth=best_max_depth,
        random_state=42
    )

    gb.fit(X_train, y_train)

    accuracy = gb.score(X_test, y_test)

    return gb, accuracy


from sklearn.neural_network import MLPClassifier

def best_mlp_hyperparameters(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_layer_sizes_list=[(50,), (100,), (100,50)],
    alpha_list=[0.0001, 0.001, 0.01]
):
    """
    Find the best hyperparameters for MLP.
    """

    best_accuracy = 0
    best_hidden_layer_sizes = None
    best_alpha = None

    for hidden_layer_sizes in hidden_layer_sizes_list:
        for alpha in alpha_list:

            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                max_iter=1000,
                random_state=42
            )

            mlp.fit(X_train, y_train)

            accuracy = mlp.score(X_val, y_val)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hidden_layer_sizes = hidden_layer_sizes
                best_alpha = alpha

    return (
        best_hidden_layer_sizes,
        best_alpha,
        best_accuracy
    )

from sklearn.neural_network import MLPClassifier

def mlp_model(
    X_train,
    y_train,
    X_test,
    y_test,
    best_hidden_layer_sizes,
    best_alpha
):
    """
    Train and evaluate an MLP classifier.
    """

    mlp = MLPClassifier(
        hidden_layer_sizes=best_hidden_layer_sizes,
        alpha=best_alpha,
        max_iter=1000,
        random_state=42
    )

    mlp.fit(X_train, y_train)

    accuracy = mlp.score(X_test, y_test)

    return mlp, accuracy


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_models(models, X_test, y_test):

    results = []

    for name, model in models.items():

        y_pred = model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred
        ).ravel()

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "False Positives": fp,
            "False Negatives": fn
        })

    return pd.DataFrame(results).sort_values(
        by="F1-score",
        ascending=False
    )


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_confusion_matrices(models, X_test, y_test):
    """
    Plot confusion matrices for multiple models.

    Parameters:
    - models: dictionary {name: trained_model}
    - X_test: test features
    - y_test: test labels
    """

    n_models = len(models)

    cols = 3
    rows = math.ceil(n_models / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4 * rows)
    )

    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):

        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i]
        )

        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    # delete empty figures
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
