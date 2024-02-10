from typing import Callable, TypeAlias
from utils import setting
from utils.stdout import StdOut

from intelligenes import selection, classification, intelligenes

PipelineResult: TypeAlias = tuple[
    str, setting.Config, Callable[[str, str, StdOut], None]
]


def feature_selection_pipeline() -> PipelineResult:
    config = setting.Config(
        [
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

    def run(
        cgit_file: str,
        output_dir: str,
        stdout: StdOut,
    ):
        selection.main(
            stdout=stdout,
            cgit_file=cgit_file,
            output_dir=output_dir,
            rand_state=config.get("Random State"),
            test_size=config.get("Test Size"),
            use_normalization=config.get("Normalize"),
            use_rfe=config.get("Recursive Feature Elimination"),
            use_pearson=config.get("Pearson's Correlation"),
            use_anova=config.get("Analysis of Variance"),
            use_chi2=config.get("Chi-Squared Test"),
        )

    return ("Feature Selection", config, run)


def classification_pipeline() -> PipelineResult:
    config = setting.Config(
        [
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

    def run(
        cgit_file: str,
        output_dir: str,
        stdout: StdOut,
    ):
        classification.main(
            stdout=stdout,
            selected_cgit_file=cgit_file,
            output_dir=output_dir,
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

    return ("Feature Classification", config, run)


def select_and_classify_pipeline() -> PipelineResult:
    config = setting.Config(
        [
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

    def run(
        cgit_file: str,
        output_dir: str,
        stdout: StdOut,
    ):
        intelligenes.main(
            cgit_file=cgit_file,
            stdout=stdout,
            output_dir=output_dir,
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

    return ("Selection and Classification", config, run)
