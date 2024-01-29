from typing import Callable
from utils import setting

from PySide6.QtCore import SignalInstance

from intelligenes import selection, classification, intelligenes


def feature_selection_pipeline(
    changeDirSignal: SignalInstance,
) -> list[tuple[str, list[setting.Setting], Callable[[], None]]]:
    inputs: list[setting.Setting] = [
        setting.CSVSetting("CIGT File", None),
        setting.DirectorySetting("Output", None),
        setting.IntSetting("Random State", 42, min=0, max=100, step=1),
        setting.FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05),
        setting.BoolSetting("Normalize", False),
        setting.BoolSetting("Recursive Feature Elimination", True),
        setting.BoolSetting("Pearson's Correlation", True),
        setting.BoolSetting("Analysis of Variance", True),
        setting.BoolSetting("Chi-Squared Test", True),
    ]

    def run():
        if inputs[0].value is not None and inputs[1].value is not None:
            selection.main(
                cgit_file=inputs[0].value,
                output_dir=inputs[1].value,
                rand_state=inputs[2].value,
                test_size=inputs[3].value,
                use_normalization=inputs[4].value,
                use_rfe=inputs[5].value,
                use_pearson=inputs[6].value,
                use_anova=inputs[7].value,
                use_chi2=inputs[8].value,
            )
            changeDirSignal.emit(inputs[1].value)

    return ("Feature Selection", inputs, run)


def classification_pipeline(
    changeDirSignal: SignalInstance,
) -> list[tuple[str, list[setting.Setting], Callable[[], None]]]:
    inputs: list[setting.Setting] = [
        setting.CSVSetting("CIGT File", None),
        setting.CSVSetting("Selected Features", None),
        setting.DirectorySetting("Output", None),
        setting.IntSetting("Random State", 42, min=0, max=100, step=1),
        setting.FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05),
        setting.IntSetting("N Splits", 5, min=1, max=20, step=1),
        setting.BoolSetting("Normalize", False),
        setting.BoolSetting("Tune", False),
        setting.StrChoiceSetting("Voting", "soft", ["soft", "hard"]),
        setting.BoolSetting("Calculate I-Genes", True),
        setting.BoolSetting("Create Visualizations", True),
        setting.BoolSetting("Random Forest", True),
        setting.BoolSetting("Support Vector Machine", True),
        setting.BoolSetting("XGBoost", True),
        setting.BoolSetting("K-Nearest Neighbors", True),
        setting.BoolSetting("Multi-Layer Perceptron", True),
    ]

    def run():
        if (
            inputs[0].value is not None
            and inputs[1].value is not None
            and inputs[2] is not None
        ):
            classification.main(
                cgit_file=inputs[0].value,
                features_file=inputs[1].value,
                output_dir=inputs[2].value,
                rand_state=inputs[3].value,
                test_size=inputs[4].value,
                n_splits=inputs[5].value,
                use_normalization=inputs[6].value,
                use_tuning=inputs[7].value,
                voting_type=inputs[8].value,
                use_igenes=inputs[9].value,
                use_visualizations=inputs[10].value,
                use_rf=inputs[11].value,
                use_svm=inputs[12].value,
                use_xgb=inputs[13].value,
                use_knn=inputs[14].value,
                use_mlp=inputs[15].value,
            )

            changeDirSignal.emit(inputs[1].value)

    return ("Feature Classification", inputs, run)


def select_and_classify_pipeline(
    changeDirSignal: SignalInstance,
) -> list[tuple[str, list[setting.Setting], Callable[[], None]]]:
    inputs: list[setting.Setting] = [
        setting.CSVSetting("CIGT File", None),
        setting.DirectorySetting("Output", None),
        setting.IntSetting("Random State", 42, min=0, max=100, step=1),
        setting.FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05),
        setting.BoolSetting("Normalize", False),
        setting.BoolSetting("Recursive Feature Elimination", True),
        setting.BoolSetting("Pearson's Correlation", True),
        setting.BoolSetting("Analysis of Variance", True),
        setting.BoolSetting("Chi-Squared Test", True),
        setting.IntSetting("N Splits", 5, min=1, max=20, step=1),
        setting.BoolSetting("Tune", False),
        setting.StrChoiceSetting("Voting", "soft", ["soft", "hard"]),
        setting.BoolSetting("Calculate I-Genes", True),
        setting.BoolSetting("Create Visualizations", True),
        setting.BoolSetting("Random Forest", True),
        setting.BoolSetting("Support Vector Machine", True),
        setting.BoolSetting("XGBoost", True),
        setting.BoolSetting("K-Nearest Neighbors", True),
        setting.BoolSetting("Multi-Layer Perceptron", True),
    ]

    def run():
        if inputs[0].value is not None and inputs[1].value is not None:
            intelligenes.main(
                cgit_file=inputs[0].value,
                output_dir=inputs[1].value,
                rand_state=inputs[2].value,
                test_size=inputs[3].value,
                use_normalization=inputs[4].value,
                use_rfe=inputs[5].value,
                use_pearson=inputs[6].value,
                use_anova=inputs[7].value,
                use_chi2=inputs[8].value,
                n_splits=inputs[9].value,
                use_tuning=inputs[10].value,
                voting_type=inputs[11].value,
                use_igenes=inputs[12].value,
                use_visualizations=inputs[13].value,
                use_rf=inputs[14].value,
                use_svm=inputs[15].value,
                use_xgb=inputs[16].value,
                use_knn=inputs[17].value,
                use_mlp=inputs[18].value,
            )
            
            changeDirSignal.emit(inputs[1].value)

    return ("Selection and Classification", inputs, run)
