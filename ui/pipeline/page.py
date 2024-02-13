# UI libraries
from PySide6.QtCore import SignalInstance
from PySide6.QtWidgets import QVBoxLayout, QPushButton, QComboBox

# Custom UI libraries
from .controls import PipelineControls
from .console import PipelineConsole
from ui.components.page import Page

# Custom utilities
from utils.capture_output import CaptureOutput
from utils.stdout import StdOut

# Intelligenes pipelines
from intelligenes.intelligenes_pipelines import (
    select_and_classify_pipeline,
    classification_pipeline,
    feature_selection_pipeline,
    PipelineResult,
)


class PipelinePage(Page):
    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ) -> None:
        super().__init__(inputFile, outputDir, onTabSelected)

        self.stdout = StdOut()
        self.output = CaptureOutput(self.stdout)

        self.inputFilePath = None
        self.outputDirPath = None

        self.inputFileSignal.connect(self._setFile)
        self.outputDirSignal.connect(self._setDir)

        pipelines: list[PipelineResult] = [
            select_and_classify_pipeline(),
            feature_selection_pipeline(),
            classification_pipeline(),
        ]

        layout = QVBoxLayout()
        self.setLayout(layout)

        combo_box = QComboBox()
        run_button = QPushButton("Run")

        for name, _, _ in pipelines:
            combo_box.addItem(name)
        combo_box.setCurrentIndex(0)

        console = PipelineConsole()

        self.output.textChanged.connect(console.setText)
        self.output.started.connect(lambda: run_button.setDisabled(True))
        self.output.finished.connect(lambda: run_button.setDisabled(False))

        run_button.clicked.connect(
            # The callback is necessary to re-emit the output directory after the process is done
            # This allows any slots listening to the signal to update with the contents of the directory
            # after it is finished
            lambda: self.run(pipelines[combo_box.currentIndex()])
        )

        controls = PipelineControls(pipelines, run_button, combo_box)

        layout.addWidget(controls)
        layout.setStretch(0, 1)
        layout.addWidget(console)
        layout.setStretch(1, 2)

    def run_pipeline(self, pipeline: PipelineResult):
        # validate pipeline
        if not self.inputFilePath:
            self.stdout.write("Select input CIGT file")
        elif not self.outputDirPath:
            self.stdout.write("Select output directory")
        else:
            pipeline[2](self.inputFilePath, self.outputDirPath, self.stdout)

    # Run process in a separate thread and capture output for the console
    def run(self, pipeline: PipelineResult):
        self.stdout.open()
        self.output.load_job(lambda: self.run_pipeline(pipeline))
        self.output.start()  # closes stdout when finished (need to reopen)

    def _setFile(self, text: str):
        self.inputFilePath = text

    def _setDir(self, text: str):
        self.outputDirPath = text
