import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout

# Custom Components
from components.tab_bar import TabBar
from pages.pipeline import PipelinePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IntelliGenes")

        layout = QVBoxLayout()

        tabs = TabBar([("Pipeline", PipelinePage())])
        tabs.setLayout(layout)

        self.setCentralWidget(tabs)


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
