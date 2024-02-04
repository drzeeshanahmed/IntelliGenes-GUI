import matplotlib as mlp

# Non-interactive backend so that file saving doesn't consume too much memory (no need for mlp GUI)
mlp.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier

# SHAP scores
from shap import KernelExplainer, TreeExplainer, LinearExplainer, sample, summary_plot
from shap.maskers import Independent

# Misc
from datetime import datetime
import os
from pathlib import Path
from typing import Any

from utils.queue import StdOut


def with_tuning(classifier, rand_state, nsplits: int, parameters: dict[str, Any]):
    return GridSearchCV(
        classifier,
        param_grid=parameters,
        cv=KFold(n_splits=nsplits, shuffle=True, random_state=rand_state),
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )


def rf_classifier(
    x: DataFrame, y: Series, rand_state: int, tuning: bool, nsplits: int, stdout: StdOut
):
    stdout.write("Random Forest")
    clf = RandomForestClassifier(random_state=rand_state)
    if tuning:
        stdout.write("Tuning Hyperparameters")
        clf = with_tuning(
            classifier=clf,
            rand_state=rand_state,
            nsplits=nsplits,
            parameters={
                "max_features": np.arange(1, x.shape[1] + 1),
                "min_samples_split": np.arange(2, 11),
                "min_samples_leaf": np.concatenate([np.arange(1, 11), [100, 150]]),
            },
        )

    result = clf.fit(x, y)
    return result.best_estimator_ if tuning else result


def svm_classifier(
    x: DataFrame, y: Series, rand_state: int, tuning: bool, nsplits: int, stdout: StdOut
):
    stdout.write("Support Vector Machine")
    clf = SVC(random_state=rand_state, kernel="linear", probability=True)
    if tuning:
        stdout.write("Tuning Hyperparameters")
        clf = with_tuning(
            classifier=clf,
            rand_state=rand_state,
            nsplits=nsplits,
            parameters={
                "kernel": ["linear"],
                "gamma": [0.001, 0.01, 0.1, 1, 10],
                "C": [0.01, 0.1, 1, 10, 50, 100, 500, 1000, 5000, 10000],
            },
        )

    result = clf.fit(x, y)
    return result.best_estimator_ if tuning else result


def xgb_classifier(
    x: DataFrame, y: Series, rand_state: int, tuning: bool, nsplits: int, stdout: StdOut
):
    stdout.write("XGBoost")
    clf = XGBClassifier(random_state=rand_state, objective="binary:logistic")
    if tuning:
        stdout.write("Tuning Hyperparameters")
        clf = with_tuning(
            classifier=clf,
            rand_state=rand_state,
            nsplits=nsplits,
            parameters={
                "n_estimators": [int(x) for x in np.linspace(100, 500, 5)],
                "max_depth": [int(x) for x in np.linspace(3, 9, 4)],
                "gamma": [0.01, 0.1],
                "learning_rate": [0.001, 0.01, 0.1, 1],
            },
        )

    result = clf.fit(x, y)
    return result.best_estimator_ if tuning else result


def knn_classifier(
    x: DataFrame, y: Series, rand_state: int, tuning: bool, nsplits: int, stdout: StdOut
):
    stdout.write("K-Nearest Neighbors")
    clf = KNeighborsClassifier()
    if tuning:
        stdout.write("Tuning Hyperparameters")
        clf = with_tuning(
            classifier=clf,
            rand_state=rand_state,
            nsplits=nsplits,
            parameters={
                "leaf_size": list(range(1, 50)),
                "n_neighbors": list(range(1, min(len(x) - 1, 30) + 1)),
                "p": [1, 2],
            },
        )

    result = clf.fit(x, y)
    return result.best_estimator_ if tuning else result


def mlp_classifier(
    x: DataFrame, y: Series, rand_state: int, tuning: bool, nsplits: int, stdout: StdOut
):
    stdout.write("Multi-Layer Perceptron")
    clf = MLPClassifier(random_state=rand_state, max_iter=2000)
    if tuning:
        stdout.write("Tuning Hyperparameters")
        clf = with_tuning(
            classifier=clf,
            rand_state=rand_state,
            nsplits=nsplits,
            parameters={
                "hidden_layer_sizes": [
                    (50, 50, 50),
                    (50, 100, 50),
                    (100,),
                    (100, 100),
                    (100, 100, 100),
                ],
                "activation": ["tanh", "relu"],
                "solver": ["sgd", "adam"],
                "alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
                "learning_rate": ["constant", "adaptive"],
                "learning_rate_init": [0.001, 0.01, 0.1],
                "momentum": [0.1, 0.2, 0.5, 0.9],
            },
        )

    result = clf.fit(x, y)
    return result.best_estimator_ if tuning else result


def voting_classifier(
    x: DataFrame,
    y: DataFrame,
    voting: str,
    names: list[str],
    classifiers: list[Any],
    stdout: StdOut,
):
    stdout.write("Voting Classifier")
    return VotingClassifier(
        estimators=list(zip(names, classifiers)), voting=voting
    ).fit(x, y)


def standard_scalar(x: DataFrame) -> DataFrame:
    return pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns)


def expression_direction(*scores: list[float]):
    positive_count = sum(1 for score in scores if score > 0)
    negative_count = sum(1 for score in scores if score < 0)

    if positive_count > negative_count:
        return "Overexpressed"
    elif negative_count > positive_count:
        return "Underexpressed"
    else:
        return "Inconclusive"


def classify_features(
    X: DataFrame,
    Y: Series,
    stdout: StdOut,
    rand_state: int,
    test_size: float,
    use_normalization: bool,
    use_tuning: bool,
    nsplits: int,
    use_rf: bool,
    use_svm: bool,
    use_xgb: bool,
    use_knn: bool,
    use_mlp: bool,
    voting_type: str,  # soft or hard
    use_visualizations: bool,
    use_igenes: bool,
    output_dir: str,
    stem: str,
):
    stdout.write("Feature Classification")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x, x_t, y, y_t = train_test_split(
        X, Y, test_size=test_size, random_state=rand_state
    )

    if use_normalization:
        stdout.write("Normalizing DataFrame")
        x = standard_scalar(x)

    names = []
    classifiers = []
    model_shaps = []

    if use_rf:
        names.append("Random Forest")
        rf = rf_classifier(x, y, rand_state, use_tuning, nsplits, stdout=stdout)
        classifiers.append(rf)
        stdout.write("Calculating SHAP-scores")
        explainer = TreeExplainer(rf)
        # Random Forest, unlike the other classifiers returns a matrix of shap values for each class (0 and 1). Other classifers return only for the 1 class.
        # Since 0 and 1 are mutually exclusive, the SHAP value for getting 1 is the opposite of getting 0.
        model_shaps.append(explainer.shap_values(x_t)[1])
    if use_svm:
        names.append("Support Vector Machine")
        svm = svm_classifier(x, y, rand_state, use_tuning, nsplits, stdout=stdout)
        classifiers.append(svm)
        stdout.write("Calculating SHAP-scores")
        explainer = LinearExplainer(svm, masker=Independent(x))
        model_shaps.append(explainer.shap_values(x_t))
    if use_xgb:
        names.append("XGBoost")
        xgb = xgb_classifier(x, y, rand_state, use_tuning, nsplits, stdout=stdout)
        classifiers.append(xgb)
        stdout.write("Calculating SHAP-scores")
        explainer = TreeExplainer(xgb)
        model_shaps.append(explainer.shap_values(x_t))
    if use_knn:
        names.append("K-Nearest Neighbors")
        knn = knn_classifier(x, y, rand_state, use_tuning, nsplits, stdout=stdout)
        classifiers.append(knn)
        stdout.write("Calculating SHAP-scores")
        explainer = KernelExplainer(knn.predict, sample(x, 1000))
        model_shaps.append(explainer.shap_values(x_t))
    if use_mlp:
        names.append("Multi-Layer Perceptron")
        mlp = mlp_classifier(x, y, rand_state, use_tuning, nsplits, stdout=stdout)
        classifiers.append(mlp)
        stdout.write("Calculating SHAP-scores")
        explainer = KernelExplainer(mlp.predict, sample(x, 1000))
        model_shaps.append(explainer.shap_values(x_t))

    if not classifiers:
        stdout.write("No classifiers were executed. Exiting...")
        return

    names.append("Voting Classifier")
    voting = voting_classifier(x, y, voting_type, names, classifiers, stdout=stdout)
    classifiers.append(voting)

    stdout.write("Calculating Metrics")
    metrics = None
    for name, classifier in zip(names, classifiers):
        y_pred = classifier.predict(x_t)
        # first column = probablity of class being 0, second column = proba of being 1
        y_prob = classifier.predict_proba(x_t)[:, 1]
        df = DataFrame(
            [
                {
                    "Classifier": name,
                    "Accuracy": accuracy_score(y_t, y_pred),
                    "ROC-AUC": roc_auc_score(y_t, y_prob),
                    "F1": f1_score(y_t, y_pred, average="weighted"),
                }
            ]
        )
        metrics = df if metrics is None else pd.concat([metrics, df], ignore_index=True)

    metrics_path = os.path.join(output_dir, f"{stem}_Classifier-Metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    stdout.write(f"Saved classifier metrics to {metrics_path}")

    if use_igenes:
        stdout.write("Calculating I-Genes score")
        feature_hhi_weights = []  # weights of each feature *per model*
        normalized_features_importances = []  # importances for each feature *per model*

        for importances in model_shaps:
            # importances has dimensionality of [samples x features]. We want [1 x features]
            # so `importances` below is a row vector of dimension `features`
            importances = np.mean(
                importances, axis=0
            )  # 'flatten' the rows into their mean
            normalized = importances / np.max(
                np.abs(importances)
            )  # scale between -1 and 1
            normalized_features_importances.append(normalized)
            # TODO: decide if axis=0 is necessary
            feature_hhi_weights.append(
                np.sum(np.square(np.abs(normalized) * 100), axis=0)
            )

        # np.sum() sums up all entries individually ([models x features] --> number) [[1, 2], [3, 4]] --> 10
        feature_hhi_weights = np.array(feature_hhi_weights) / np.sum(
            feature_hhi_weights
        )
        initial_weights = 1 / len(classifiers)
        final_weights = initial_weights + (initial_weights * feature_hhi_weights)

        # Should become a list of dimension features
        igenes_scores = None
        for weight, importance in zip(final_weights, normalized_features_importances):
            result = weight * np.abs(importance)
            igenes_scores = (
                igenes_scores + result if igenes_scores is not None else result
            )

        # of size [features x models] --> list(zip(*)) converts a list of lists to a list of tuples that is the transpose
        transposed_importances = list(zip(*normalized_features_importances))
        # sums the expression direction of a featuer per model
        directions = [
            expression_direction(*scores) for scores in transposed_importances
        ]

        igenes_df = DataFrame(
            {
                "Features": x_t.columns,
                "I-Genes Score": igenes_scores,
                "Expression Direction": directions,
            }
        )
        igenes_df["I-Genes Rankings"] = (
            igenes_df["I-Genes Score"].rank(ascending=False).astype(int)
        )

        igenes_path = os.path.join(output_dir, f"{stem}_I-Genes-Score.csv")
        igenes_df.to_csv(igenes_path, index=False)
        stdout.write(f"Saved igenes scores to {igenes_path}")

    if use_visualizations:
        stdout.write("Generating visualizations")

        for name, importances in zip(names, model_shaps):
            stdout.write(f"Generating summary_plot for {name}")

            summary_plot(importances, x_t, plot_type="dot", show=False)
            plt.title(f"{name} Feature Importances", fontsize=16)
            plt.xlabel("SHAP Value", fontsize=14)
            plt.ylabel("Feature", fontsize=14)
            plt.tight_layout()

            plot_path = os.path.join(
                output_dir, f"{stem}_{name.replace(' ', '-')}-SHAP.png"
            )
            plt.savefig(plot_path)
            plt.clf()
            plt.close()

    stdout.write("Finished Feature Classification")


def main(
    stdout: StdOut,
    cgit_file: str,
    features_file: str,
    output_dir: str,
    rand_state: int,
    test_size: float,
    n_splits: int,
    voting_type: str,
    use_tuning: bool,
    use_normalization: bool,
    use_igenes: bool,
    use_visualizations: bool,
    use_rf: bool,
    use_svm: bool,
    use_xgb: bool,
    use_knn: bool,
    use_mlp: bool,
):
    # Same as what is outputed from `selection.py`
    selected_column = "Features"
    y_label_col = "Type"

    stdout.write(f"Reading Data from {cgit_file}")
    input_df = pd.read_csv(cgit_file).drop(columns=["ID"])
    stdout.write(f"Reading Selected Features from {features_file}")
    selected = pd.read_csv(features_file)[selected_column].values.flatten().tolist()

    X = input_df[selected]
    Y = input_df[y_label_col]

    classify_features(
        X,
        Y,
        rand_state=rand_state,
        test_size=test_size,
        use_normalization=use_normalization,
        use_tuning=use_tuning,
        nsplits=n_splits,
        use_rf=use_rf,
        use_svm=use_svm,
        use_xgb=use_xgb,
        use_knn=use_knn,
        use_mlp=use_mlp,
        voting_type=voting_type,
        use_visualizations=use_visualizations,
        use_igenes=use_igenes,
        output_dir=output_dir,
        stem=f"{Path(cgit_file).stem}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}",
        stdout=stdout,
    )
