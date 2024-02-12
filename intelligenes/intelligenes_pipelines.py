# Miscellaneous System libraries
from typing import Callable, TypeAlias

# Custom utilities
from utils.setting import (
    Config,
    Group,
    BoolSetting,
    IntSetting,
    FloatSetting,
    StrChoiceSetting,
)
from utils.stdout import StdOut

# IntelliGenes pipelines
from . import selection, classification, intelligenes

# Custom type alias
PipelineResult: TypeAlias = tuple[str, Config, Callable[[str, str, StdOut], None]]


def feature_selection_pipeline() -> PipelineResult:
    config = Config(
        [
            Group(
                "Parameters",
                [
                    IntSetting(
                        "Random State",
                        42,
                        min=0,
                        max=100,
                        step=1,
                        tooltip="Random state can be set to create \nreproducibile results across runs.",
                    ),
                    FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05, tooltip=""),
                    BoolSetting("Normalize", False, tooltip=""),
                ],
            ),
            Group(
                "Selectors",
                [
                    BoolSetting("Recursive Feature Elimination", True, tooltip=""),
                    BoolSetting("Pearson's Correlation", True, tooltip=""),
                    BoolSetting("Analysis of Variance", True, tooltip=""),
                    BoolSetting("Chi-Squared Test", True, tooltip=""),
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
    config = Config(
        [
            Group(
                "Parameters",
                [
                    IntSetting("Random State", 42, min=0, max=100, step=1, tooltip=""),
                    FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05, tooltip=""),
                    IntSetting("N Splits", 5, min=1, max=20, step=1, tooltip=""),
                    BoolSetting("Normalize", False, tooltip=""),
                    BoolSetting("Tune", False, tooltip=""),
                    StrChoiceSetting("Voting", "soft", ["soft", "hard"], tooltip=""),
                    BoolSetting("Calculate I-Genes", True, tooltip=""),
                    BoolSetting("Create Visualizations", True, tooltip=""),
                ],
            ),
            Group(
                "Classifiers",
                [
                    BoolSetting("Random Forest", True, tooltip=""),
                    BoolSetting("Support Vector Machine", True, tooltip=""),
                    BoolSetting("XGBoost", True, tooltip=""),
                    BoolSetting("K-Nearest Neighbors", True, tooltip=""),
                    BoolSetting("Multi-Layer Perceptron", True, tooltip=""),
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
    config = Config(
        [
            Group(
                "Parameters",
                [
                    IntSetting("Random State", 42, min=0, max=100, step=1, tooltip=""),
                    FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05, tooltip=""),
                    BoolSetting("Normalize", False, tooltip=""),
                    IntSetting("N Splits", 5, min=1, max=20, step=1, tooltip=""),
                    BoolSetting("Tune", False, tooltip=""),
                    StrChoiceSetting("Voting", "soft", ["soft", "hard"], tooltip=""),
                    BoolSetting("Calculate I-Genes", True, tooltip=""),
                    BoolSetting("Create Visualizations", True, tooltip=""),
                ],
            ),
            Group(
                "Selectors",
                [
                    BoolSetting("Recursive Feature Elimination", True, tooltip=""),
                    BoolSetting("Pearson's Correlation", True, tooltip=""),
                    BoolSetting("Analysis of Variance", True, tooltip=""),
                    BoolSetting("Chi-Squared Test", True, tooltip=""),
                ],
            ),
            Group(
                "Classifiers",
                [
                    BoolSetting("Random Forest", True, tooltip=""),
                    BoolSetting("Support Vector Machine", True, tooltip=""),
                    BoolSetting("XGBoost", True, tooltip=""),
                    BoolSetting("K-Nearest Neighbors", True, tooltip=""),
                    BoolSetting("Multi-Layer Perceptron", True, tooltip=""),
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
