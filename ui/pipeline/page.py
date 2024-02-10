from typing import Callable
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox
from PySide6.QtCore import SignalInstance

from .controls import PipelineControls
from .console import PipelineConsole
from utils.capture_output import CaptureOutput
from utils.stdout import StdOut

from intelligenes.intelligenes_pipelines import (
    select_and_classify_pipeline,
    classification_pipeline,
    feature_selection_pipeline,
    PipelineResult,
)


class PipelinePage(QWidget):
    def __init__(
        self, inputFileSignal: SignalInstance, outputDirSignal: SignalInstance
    ) -> None:
        super().__init__()

        self.stdout = StdOut()
        self.output = CaptureOutput(self.stdout)

        self.inputFile = None
        self.outputDir = None

        inputFileSignal.connect(self._setFile)
        outputDirSignal.connect(self._setDir)

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
            lambda: self.run(
                pipelines[combo_box.currentIndex()],
                lambda: outputDirSignal.emit(self.outputDir),
            )
        )

        controls = PipelineControls(pipelines, run_button, combo_box)

        layout.addWidget(controls)
        layout.setStretch(0, 1)
        layout.addWidget(console)
        layout.setStretch(1, 2)

    def run_pipeline(self, pipeline: PipelineResult, callback: Callable[[], None]):
        # validate pipeline
        if not self.inputFile:
            self.stdout.write("Select input CIGT file")
        elif not self.outputDir:
            self.stdout.write("Select output directory")
        else:
            pipeline[2](self.inputFile, self.outputDir, self.stdout)
            callback()

    # Run process in a separate thread and capture output for the console
    def run(self, pipeline: PipelineResult, callback: Callable[[], None]):
        self.stdout.open()
        self.output.load_job(lambda: self.run_pipeline(pipeline, callback))
        self.output.start()  # closes stdout when finished (need to reopen)

    def _setFile(self, text: str):
        self.inputFile = text

    def _setDir(self, text: str):
        self.outputDir = text
