from PySide6.QtWidgets import QTabWidget, QWidget

class TabBar(QTabWidget):
    def __init__(self, widgets: list[tuple[str, QWidget]]) -> None:
        super().__init__()
        self.setTabPosition(QTabWidget.North)
        self.setDocumentMode(True)

        for name, widget in widgets:
            self.addTab(widget, name)