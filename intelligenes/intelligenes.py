# Data Tools
import pandas as pd

# Intelligenes
from selection import select_features
from classification import classify_features

# Misc
import argparse
from datetime import datetime
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--cgit_file", required=True)
    parser.add_argument("-o", "--output_dir", required=True)

    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--n_splits", type=int, default=5)

    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--no_rfe", action="store_true")
    parser.add_argument("--no_pearson", action="store_true")
    parser.add_argument("--no_chi2", action="store_true")
    parser.add_argument("--no_anova", action="store_true")

    parser.add_argument("--voting", type=str, default="soft")
    parser.add_argument("--no_rf", action="store_true")
    parser.add_argument("--no_svm", action="store_true")
    parser.add_argument("--no_xgb", action="store_true")
    parser.add_argument("--no_knn", action="store_true")
    parser.add_argument("--no_mlp", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--no_igenes", action="store_true")
    parser.add_argument("--no_visualizations", action="store_true")

    args = parser.parse_args()

    y_label_col = "Type"
    output_features_col = "Features"

    print(f"Reading DataFrame from {args.cgit_file}")

    input_df = pd.read_csv(args.cgit_file).drop(columns=["ID"])
    X = input_df.drop(columns=[y_label_col])
    Y = input_df[y_label_col]

    selected = select_features(
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

    X = input_df[selected]
    Y = input_df[y_label_col]
    classify_features(
        X,
        Y,
        rand_state=args.random_state,
        test_size=args.test_size,
        use_normalization=args.normalize,
        use_tuning=args.tune,
        nsplits=args.n_splits,
        use_rf=not args.no_rf,
        use_svm=not args.no_svm,
        use_xgb=not args.no_xgb,
        use_knn=not args.no_knn,
        use_mlp=not args.no_mlp,
        voting_type=args.voting,
        use_visualizations=not args.no_visualizations,
        use_igenes=not args.no_igenes,
        output_dir=args.output_dir,
        stem=f"{Path(args.cgit_file).stem}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}",
    )

    print("Finished Feature Classification")
    print("Finished Intelligenes Pipeline")
