import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTabWidget

# Custom Components
from ui.files.page import FilesPage
from ui.pipeline.page import PipelinePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IntelliGenes")

        layout = QVBoxLayout()
        
        files = FilesPage()
        pipeline = PipelinePage(changeDirSignal=files.selectedDir)
        tabs = [("Pipeline", pipeline), ("Files", files)]

        tab_bar = QTabWidget()
        tab_bar.setTabPosition(QTabWidget.TabPosition.North)
        tab_bar.setDocumentMode(True)

        for name, widget in tabs:
            tab_bar.addTab(widget, name)
        tab_bar.setLayout(layout)

        self.setCentralWidget(tab_bar)


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
