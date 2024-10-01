# packages
import time
import pathlib
import random
import numpy as np
import pandas as pd
from pprint import pprint
import humanfriendly as human

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import ohtconf as conf

def make_strlabel(ay: pd.Series, intlabels: list[int] | None = None, strlabels: list[str] | None = None) -> pd.Series:
    """map int labels to str labels

    map values (0~10) to (NORM,TEM,...,CT3)
    """
    if intlabels is None or strlabels is None:
        all_columns = conf.COLUMN_GRAPH
        intlabels = list(range(0, 11))  # 0 ~ 10 for normal, outlier-tem,...,outlier-ct4 flag
        strlabels = ["NORM"] + all_columns  # NORM for normal, others are for outlier

    label_dict = {i: s for i, s in zip(intlabels, strlabels)}
    ay = ay.map(label_dict)
    return ay


def make_intlabel(ay: pd.Series, strlabels: list[str] | None = None, intlabels: list[int] | None = None) -> pd.Series:
    """map str labels to int labels

    map values (NORM,TEM,...,CT3) to (0~10)
    """
    if intlabels is None or strlabels is None:
        all_columns = conf.COLUMN_GRAPH
        intlabels = list(range(0, 11))  # 0 ~ 10 for normal, outlier-tem,...,outlier-ct4 flag
        strlabels = ["NORM"] + all_columns  # NORM for normal, others are for outlier

    label_dict = {s: i for s, i in zip(strlabels, intlabels)}
    ay = ay.map(label_dict)
    return ay


def calc_neighbors(
    ay: pd.Series, minv: int | None = None, maxv: int | None = None, countv: int | None = None
) -> list[int]:
    """calc neighbors candidates based on target class count"""
    nunique = ay.nunique()
    lower = nunique if minv is None else minv
    upper = nunique**2 if maxv is None else maxv
    count = min(conf.NEIGHBORS_BASE, upper - lower) if countv is None else min(countv, upper - lower)

    random.seed(42)  # reproducable for the same input
    neighbors = sorted(random.sample(range(lower, upper + 1), count))

    assert len(neighbors) > 0, f"calc_neighbors invalid neighbors={neighbors}, minv={minv},maxv={maxv},countv={countv}"
    # print(f"calc neighbors count={len(neighbors)}, value={neighbors}")
    return neighbors


def train(
    X: pd.DataFrame, y: pd.Series, columns: list[str], labels: list[str], neighbors: list[int] | None = None
) -> tuple[int, KNeighborsClassifier, pd.DataFrame, pd.Series]:
    """knn modle training"""

    def do_x_scale(ax: pd.DataFrame, acolumns: list[str]) -> pd.DataFrame:
        """standardize feature values"""
        scaler = StandardScaler()
        ax = pd.DataFrame(scaler.fit_transform(ax), columns=acolumns)
        return ax

    def do_grid_search(xtrain: pd.DataFrame, ytrain: pd.Series, aneighbors: list[int]) -> GridSearchCV:
        """grid search for the best k value"""
        _start = time.time()
        knn = KNeighborsClassifier()  # Create KNN classifier

        param_grid = {"n_neighbors": aneighbors}  # Grid search for optimal K value
        print(f"param_grid={param_grid}")

        # K-fold Cross-Validation: Divide the dataset into K folds, cv.
        # Train the model on K-1 folds and evaluate on the remaining fold. Repeat this K times, rotating the validation fold.
        gsearch = GridSearchCV(estimator=knn, param_grid=param_grid, cv=4, verbose=3)

        # verbose 0 (default): No output is printed.
        # verbose 1: The number of iterations, the current score, and the best score so far are printed.
        # verbose 2: The same information as verbose=1, plus the parameters of the current model being evaluated.
        # verbose 3: The same information as verbose=2, plus the scores for each cross-validation fold.
        gsearch.fit(xtrain, ytrain)

        # Print the best score and best estimator
        print("Best score:", gsearch.best_score_)
        print("Best params:", gsearch.best_params_)
        print("Cross Validation results:")
        pprint(gsearch.cv_results_)

        _elapsed = time.time() - _start
        print(f"do_grid_search elapsed time: {human.format_timespan(_elapsed)}")
        return gsearch

    def do_knn_train(xtrain: pd.DataFrame, ytrain: pd.Series, best_k: int) -> KNeighborsClassifier:
        _start = time.time()
        # model with best K
        amodel = KNeighborsClassifier(n_neighbors=best_k)
        amodel.fit(xtrain, ytrain)
        _elapsed = time.time() - _start
        print(f"do_knn_train elapsed time: {human.format_timespan(_elapsed)}")
        return amodel

    #
    _start = time.time()

    # x scaling
    if conf.USESCALE:
        X = do_x_scale(X, columns)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 20% test set

    # search best_k by using grid serarch
    if neighbors is None:
        neighbors = calc_neighbors(y)

    gridsearch = do_grid_search(X_train, y_train, neighbors)

    best_k = gridsearch.best_params_["n_neighbors"]

    # knn train with best_k
    model = do_knn_train(X_train, y_train, best_k)

    _elapsed = time.time() - _start
    print(f"knn_train elapsed time: {human.format_timespan(_elapsed)}")

    return best_k, model, X_test, y_test


def predict(model: KNeighborsClassifier, xtest: pd.DataFrame) -> pd.Series:
    # predictions
    ypred = model.predict(xtest)
    return ypred


def report(bestk: int, ytest: pd.Series, ypred: pd.Series, labels: list[str], title: str) -> pd.DataFrame:
    print(f"\n*** Start of KNN modeling report for {title} ***\n")

    # best_k
    print(f"\nBest K = {bestk}")

    # Accuracy score = (TP + TN) / (TP + TN + FP + FN)
    accuracy = accuracy_score(ytest, ypred)
    print("\nAccuracy Score = ", accuracy)

    # Confusion Matrix
    #                   Predicted Positive	Predicted Negative
    # Actual Positive	True Positive (TP)	False Negative (FN)
    # Actual Negative	False Positive (FP)	True Negative (TN)
    cm = confusion_matrix(ytest, ypred, labels=labels)
    dfcm = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nConfusion Matrix :\n", dfcm)

    # Classification Report
    # Precisoin = TP / (TP + FP)
    # Recall (Sensitivity) = TP / (TP + FN)
    # F1 Score = 2 * (Precision x Recall) / (Precision + Recall)
    # Support - instances of class in the dataset

    report = classification_report(ytest, ypred, labels=labels)
    print("\nClassification Report :\n", report)

    print(f"\n*** End of KNN modeling report for {title} ***\n")

    return dfcm


def cm_heatmap(dfcm: pd.DataFrame, title: str, pngfile: str) -> None:
    """knn confusion matrix heatmap

    A confusion matrix helps you understand the performance of the classification algorithm by showing the true vs. predicted labels
    """
    fig, ox = plt.subplots(nrows=1, ncols=1, figsize=conf.PLOTSIZE)
    sns.heatmap(dfcm, annot=True, cmap="coolwarm", linewidths=0.5, fmt="d", ax=ox)
    ox.xaxis.tick_top()
    ox.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ox.set_xlabel("Predicted", labelpad=15)
    ox.set_ylabel("Actual")

    fig.suptitle(f"Confusion Matrix of {title}")

    plt.tight_layout()
    plt.show()

    if pngfile is not None:
        pngfile = pngfile.lower()
        filepath = pathlib.Path(conf.DIRCHART) / pngfile
        fig.savefig(
            filepath,
            dpi=conf.DPI,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=None,
            transparent=False,
            bbox_inches=None,
            pad_inches=None,
        )


def f2_boundary_scatter(
    ax: pd.DataFrame, ay: pd.Series, amodel: KNeighborsClassifier, bestk: int, title: str, pngfile: str
) -> None:
    """knn 2 features decison boundary scatter

    This plot shows how the KNN algorithm classifies different regions of the feature space.
    """

    assert len(ax.columns) == 2, "knn_2fs_db_scatter arg1, ax should have 2 features"

    # Create color maps
    light_colors = [
        "#F08080",
        "#FFDAB9",
        "#98FB98",
        "#87CEFA",
        "#E6E6FA",
        "#FFE4E1",
        "#FFFACD",
        "#F0FFF0",
        "#F0F8FF",
        "#D8BFD8",
        "#FFA07A",
        "#FFB6C1",
        "#AFEEEE",
        "#FAFAD2",
        "#FFF0F5",
        "#FFF5EE",
        "#F5FFFA",
        "#F0FFFF",
        "#F5F5DC",
        "#B0C4DE",
    ]

    bold_colors = [
        "#DC143C",
        "#FF8C00",
        "#228B22",
        "#4169E1",
        "#9400D3",
        "#B22222",
        "#DAA520",
        "#3CB371",
        "#1E90FF",
        "#8B008B",
        "#FF6347",
        "#008B8B",
        "#9370DB",
        "#4682B4",
        "#8B0000",
        "#FF4500",
        "#2E8B57",
        "#6A5ACD",
        "#4B0082",
        "#D2691E",
    ]

    assert ay.nunique() <= len(
        light_colors
    ), f"knn_2fs_db_scatter arg2, ay nunique={ay.nunique()} lte colors={len(light_colors)}"

    cmap_light = ListedColormap(light_colors)
    cmap_bold = ListedColormap(bold_colors)

    # Create a mesh grid for plotting decision boundaries
    x_add = 10 if ax.iloc[:, 0].max() / 100 >= 1 else 1
    y_add = 10 if ax.iloc[:, 1].max() / 100 >= 1 else 1
    x_min, x_max = ax.iloc[:, 0].min() - x_add, ax.iloc[:, 0].max() + x_add
    y_min, y_max = ax.iloc[:, 1].min() - y_add, ax.iloc[:, 1].max() + y_add

    # Predict the class for each point in the mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))

    # np.c_[array,array] - translates slice objects to concatenation along the second axis. xx.ravel() - return flattened array,
    dfgrid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=ax.columns)
    # print(f"type dfgrid={type(dfgrid)}, dfgrid={dfgrid.head()}")
    Z = amodel.predict(dfgrid)  # 1-d array
    Z = pd.Series(Z)  # convert to pd.Series to use map
    Z = make_intlabel(Z)  # predicted str label map to int label for using color map at pcolormesh()
    Z = Z.to_numpy()  # convert to 1-d array
    Z = Z.reshape(xx.shape)  # reshape

    # Plot also the training points
    fig, ox = plt.subplots(nrows=1, ncols=1, figsize=conf.PLOTSIZE)

    # print(f"type xx={type(xx)}, yy={type(yy)}, Z={type(Z)}"); print(f"valuex x={xx[:3]}, yy={yy[:3]} Z={Z[:3]}")
    ox.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot the training points
    # c=y - sequence of n numbers to be mapped to colors using *cmap*
    yi = make_intlabel(ay)
    ox.scatter(ax.iloc[:, 0], ax.iloc[:, 1], c=yi, cmap=cmap_bold, edgecolor="k", s=10)
    ox.set_xlim(xx.min(), xx.max())
    ox.set_ylim(yy.min(), yy.max())
    ox.set_xlabel(ax.columns[0])
    ox.set_ylabel(ax.columns[1])

    fig.suptitle(f"K={bestk} class classification of {title}")

    plt.show()

    if pngfile is not None:
        pngfile = pngfile.lower()
        filepath = pathlib.Path(conf.DIRCHART) / pngfile
        fig.savefig(
            filepath,
            dpi=conf.DPI,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=None,
            transparent=False,
            bbox_inches=None,
            pad_inches=None,
        )
