# Data Tools
import pandas as pd

# Intelligenes
from .selection import select_features
from .classification import classify_features

# Misc
from datetime import datetime
from pathlib import Path

from utils.stdout import StdOut


def main(
    cgit_file: str,
    output_dir: str,
    rand_state: int,
    test_size: float,
    use_normalization: bool,
    use_rfe: bool,
    use_pearson: bool,
    use_anova: bool,
    use_chi2: bool,
    n_splits: int,
    voting_type: str,
    use_tuning: bool,
    use_igenes: bool,
    use_visualizations: bool,
    use_rf: bool,
    use_svm: bool,
    use_xgb: bool,
    use_knn: bool,
    use_mlp: bool,
    stdout: StdOut,
):  
    y_label_col = "Type"
    output_features_col = "Features"

    stdout.write(f"Reading DataFrame from {cgit_file}")

    input_df = pd.read_csv(cgit_file).drop(columns=["ID"])
    X = input_df.drop(columns=[y_label_col])
    Y = input_df[y_label_col]

    selected = select_features(
        X,
        Y,
        features_col=output_features_col,
        rand_state=rand_state,
        test_size=test_size,
        use_normalization=use_normalization,
        use_rfe=use_rfe,
        use_pearson=use_pearson,
        use_anova=use_anova,
        use_chi2=use_chi2,
        output_dir=output_dir,
        stem=f"{Path(cgit_file).stem}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}",
        stdout=stdout,
    )

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
        stdout=stdout
    )

    stdout.write("Finished Intelligenes Pipeline")
