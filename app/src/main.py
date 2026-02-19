# torch import necessary
import torch
from ui.app_window import KanjiWindow

def main() -> None:
    """Creates the app window and starts the application."""
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = KanjiWindow(app)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()