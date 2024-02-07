from typing import Callable
from utils import setting
from utils.stdout import StdOut

from PySide6.QtCore import SignalInstance

from intelligenes import selection, classification, intelligenes


def feature_selection_pipeline(
    changeDirSignal: SignalInstance, stdout: StdOut
) -> list[tuple[str, setting.Config, Callable[[], None]]]:
    config = setting.Config(
        [
            setting.Group(
                "Files",
                [
                    setting.CSVSetting("CIGT File", None),
                    setting.DirectorySetting("Output", None),
                ],
            ),
            setting.Group(
                "Parameters",
                [
                    setting.IntSetting("Random State", 42, min=0, max=100, step=1),
                    setting.FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05),
                    setting.BoolSetting("Normalize", False),
                ],
            ),
            setting.Group(
                "Selectors",
                [
                    setting.BoolSetting("Recursive Feature Elimination", True),
                    setting.BoolSetting("Pearson's Correlation", True),
                    setting.BoolSetting("Analysis of Variance", True),
                    setting.BoolSetting("Chi-Squared Test", True),
                ],
            ),
        ]
    )

    def run():
        if config.get("CIGT File") is None:
            stdout.write("Select a CIGT file")
        elif config.get("Output") is None:
            stdout.write("Select an output folder")
        else:
            selection.main(
                stdout=stdout,
                cgit_file=config.get("CIGT File"),
                output_dir=config.get("Output"),
                rand_state=config.get("Random State"),
                test_size=config.get("Test Size"),
                use_normalization=config.get("Normalize"),
                use_rfe=config.get("Recursive Feature Elimination"),
                use_pearson=config.get("Pearson's Correlation"),
                use_anova=config.get("Analysis of Variance"),
                use_chi2=config.get("Chi-Squared Test"),
            )
            changeDirSignal.emit(config.get("Output"))

    return ("Feature Selection", config, run)


def classification_pipeline(
    changeDirSignal: SignalInstance, stdout: StdOut
) -> list[tuple[str, setting.Config, Callable[[], None]]]:
    config = setting.Config(
        [
            setting.Group(
                "Files",
                [
                    setting.CSVSetting("CIGT File", None),
                    setting.CSVSetting("Selected Features", None),
                    setting.DirectorySetting("Output", None),
                ],
            ),
            setting.Group(
                "Parameters",
                [
                    setting.IntSetting("Random State", 42, min=0, max=100, step=1),
                    setting.FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05),
                    setting.IntSetting("N Splits", 5, min=1, max=20, step=1),
                    setting.BoolSetting("Normalize", False),
                    setting.BoolSetting("Tune", False),
                    setting.StrChoiceSetting("Voting", "soft", ["soft", "hard"]),
                    setting.BoolSetting("Calculate I-Genes", True),
                    setting.BoolSetting("Create Visualizations", True),
                ],
            ),
            setting.Group(
                "Classifiers",
                [
                    setting.BoolSetting("Random Forest", True),
                    setting.BoolSetting("Support Vector Machine", True),
                    setting.BoolSetting("XGBoost", True),
                    setting.BoolSetting("K-Nearest Neighbors", True),
                    setting.BoolSetting("Multi-Layer Perceptron", True),
                ],
            ),
        ]
    )

    def run():
        if config.get("CIGT File") is None:
            stdout.write("Select a CIGT file")
        elif config.get("Selected Features") is None:
            stdout.write("Select a significant features file")
        elif config.get("Output") is None:
            stdout.write("Select an output folder")
        else:
            classification.main(
                stdout=stdout,
                cgit_file=config.get("CIGT File"),
                features_file=config.get("Selected Features"),
                output_dir=config.get("Output"),
                rand_state=config.get("Random State"),
                test_size=config.get("Test Size"),
                n_splits=config.get("N Splits"),
                use_normalization=config.get("Normalize"),
                use_tuning=config.get("Tune"),
                voting_type=config.get("Voting"),
                use_igenes=config.get("Calculate I-Genes"),
                use_visualizations=config.get("Create Visualizations"),
                use_rf=config.get("Random Forest"),
                use_svm=config.get("Support Vector Machine"),
                use_xgb=config.get("XGBoost"),
                use_knn=config.get("K-Nearest Neighbors"),
                use_mlp=config.get("Multi-Layer Perceptron"),
            )

            changeDirSignal.emit(config.get("Output"))

    return ("Feature Classification", config, run)


def select_and_classify_pipeline(
    changeDirSignal: SignalInstance, stdout: StdOut
) -> list[tuple[str, setting.Config, Callable[[], None]]]:
    config = setting.Config(
        [
            setting.Group(
                "Files",
                [
                    setting.CSVSetting("CIGT File", None),
                    setting.DirectorySetting("Output", None),
                ],
            ),
            setting.Group(
                "Parameters",
                [
                    setting.IntSetting("Random State", 42, min=0, max=100, step=1),
                    setting.FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05),
                    setting.BoolSetting("Normalize", False),
                    setting.IntSetting("N Splits", 5, min=1, max=20, step=1),
                    setting.BoolSetting("Tune", False),
                    setting.StrChoiceSetting("Voting", "soft", ["soft", "hard"]),
                    setting.BoolSetting("Calculate I-Genes", True),
                    setting.BoolSetting("Create Visualizations", True),
                ],
            ),
            setting.Group(
                "Selectors",
                [
                    setting.BoolSetting("Recursive Feature Elimination", True),
                    setting.BoolSetting("Pearson's Correlation", True),
                    setting.BoolSetting("Analysis of Variance", True),
                    setting.BoolSetting("Chi-Squared Test", True),
                ],
            ),
            setting.Group(
                "Classifiers",
                [
                    setting.BoolSetting("Random Forest", True),
                    setting.BoolSetting("Support Vector Machine", True),
                    setting.BoolSetting("XGBoost", True),
                    setting.BoolSetting("K-Nearest Neighbors", True),
                    setting.BoolSetting("Multi-Layer Perceptron", True),
                ],
            ),
        ]
    )

    def run():
        if config.get("CIGT File") is None:
            stdout.write("Select a CIGT File")
        elif config.get("Output") is None:
            stdout.write("Select an output folder")
        else:
            intelligenes.main(
                stdout=stdout,
                cgit_file=config.get("CIGT File"),
                output_dir=config.get("Output"),
                rand_state=config.get("Random State"),
                test_size=config.get("Test Size"),
                use_normalization=config.get("Normalize"),
                use_rfe=config.get("Recursive Feature Elimination"),
                use_pearson=config.get("Pearson's Correlation"),
                use_anova=config.get("Analysis of Variance"),
                use_chi2=config.get("Chi-Squared Test"),
                n_splits=config.get("N Splits"),
                use_tuning=config.get("Tune"),
                voting_type=config.get("Voting"),
                use_igenes=config.get("Calculate I-Genes"),
                use_visualizations=config.get("Create Visualizations"),
                use_rf=config.get("Random Forest"),
                use_svm=config.get("Support Vector Machine"),
                use_xgb=config.get("XGBoost"),
                use_knn=config.get("K-Nearest Neighbors"),
                use_mlp=config.get("Multi-Layer Perceptron"),
            )

            changeDirSignal.emit(config.get("Output"))

    return ("Selection and Classification", config, run)
