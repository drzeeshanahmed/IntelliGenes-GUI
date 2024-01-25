import argparse
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr


def recursive_elim(
    x: DataFrame, y: Series, rand_state: int, features_col: str, ranking_col: str
) -> DataFrame:
    print("Recursive Feature Elimination")
    e = DecisionTreeClassifier(random_state=rand_state)
    rfe = RFE(estimator=e, n_features_to_select=1).fit(x, y)

    df = pd.DataFrame(
        {
            "features": x.columns,
            "rank": rfe.ranking_,
        },
    )
    # Since we only select 1 feature above, only that feature will have rank '1'
    # Every other feature will have a sequential rank. So here, we select the top 10% of features
    # This is done so we have an explicit order of the features.
    return df.rename(columns={"features": features_col, "rank": ranking_col})


def pearson(x: DataFrame, y: Series, features_col: str, p_value_col: str) -> DataFrame:
    print("Pearson Correlation")
    df = pd.DataFrame(
        {
            "features": x.columns,
            # Independently calculate p-value for each predictor vs y
            # the second element of the return is the p-value
            "p-value": [pearsonr(x[col], y)[1] for col in x.columns],
        }
    )

    return df.rename(columns={"features": features_col, "p-value": p_value_col})


def chi2_test(
    x: DataFrame, y: Series, features_col: str, p_value_col: str
) -> DataFrame:
    print("Chi-Squared Test")
    chi = SelectKBest(score_func=chi2, k="all").fit(x, y)
    df = pd.DataFrame(
        {
            "features": x.columns,
            "p-value": chi.pvalues_,
        }
    )

    return df.rename(columns={"features": features_col, "p-value": p_value_col})


def anova(x: DataFrame, y: Series, features_col: str, p_value_col: str) -> DataFrame:
    print("Analysis of Variance")
    anova = SelectKBest(score_func=f_classif, k="all").fit(x, y)
    df = pd.DataFrame(
        {
            "features": x.columns,
            "p-value": anova.pvalues_,
        }
    )

    return df.rename(columns={"features": features_col, "p-value": p_value_col})


def min_max_scalar(x: DataFrame) -> DataFrame:
    print("Normalizing DataFrame")
    return pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns)


### Calculates the most relevant features from `input` needed to calculate
def select_features(
    X: DataFrame,
    Y: Series,
    features_col: str,
    test_size: int,
    random_state: int,
    use_normalization: bool,
    use_rfe: bool,
    use_anova: bool,
    use_chi2: bool,
    use_pearson: bool,
    output_dir: str,
    stem: str,
):
    print("Selecting Important Features")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x, _, y, _ = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    rfe_col = "RFE Rankings"
    anova_col = "ANOVA (p-value)"
    chi2_col = "Chi-Square Test (p-value)"
    pearson_col = "Pearson's Correlation (p-value)"

    if use_normalization:
        x = min_max_scalar(x)

    results: list[DataFrame] = []
    if use_rfe:
        results.append(recursive_elim(x, y, random_state, features_col, rfe_col))
    if use_anova:
        results.append(anova(x, y, features_col, anova_col))
    if use_chi2:
        results.append(chi2_test(x, y, features_col, chi2_col))
    if use_pearson:
        results.append(pearson(x, y, features_col, pearson_col))

    all = None
    for result in results:
        all = result if all is None else all.merge(result, on=features_col)

    selected = all.loc[
        (all[rfe_col] <= int(all.shape[0] * 0.1))
        & (all[pearson_col] < 0.05)
        & (all[chi2_col] < 0.05)
        & (all[anova_col] < 0.05)
    ]

    all_path = os.path.join(output_dir, f"{stem}_All-Features.csv")
    selected_path = os.path.join(output_dir, f"{stem}_Selected-Features.csv")

    all.to_csv(all_path, index=False)
    selected.to_csv(selected_path, index=False)
    print(f"Saved all feature rankings to {all_path}")
    print(f"Saved selected feature rankings to {selected_path}")
    
    return selected[features_col]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--cgit_file", required=True)
    parser.add_argument("-o", "--output_dir", required=True)

    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--no_rfe", action="store_true")
    parser.add_argument("--no_pearson", action="store_true")
    parser.add_argument("--no_chi2", action="store_true")
    parser.add_argument("--no_anova", action="store_true")

    args = parser.parse_args()

    y_label_col = "Type"
    output_features_col = "Features"

    print(f"Reading DataFrame from {args.cgit_file}")

    input_df = pd.read_csv(args.cgit_file).drop(columns=["ID"])
    X = input_df.drop(columns=[y_label_col])
    Y = input_df[y_label_col]

    select_features(
        X,
        Y,
        features_col=output_features_col,
        random_state=args.random_state,
        test_size=args.test_size,
        use_normalization=args.normalize,
        use_rfe=not args.no_rfe,
        use_pearson=not args.no_pearson,
        use_anova=not args.no_anova,
        use_chi2=not args.no_chi2,
        output_dir=args.output_dir,
        stem=f"{Path(args.cgit_file).stem}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}",
    )

    print("Finished Feature Selection")
