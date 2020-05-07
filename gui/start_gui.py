"""
Starts the GUI, uses PyQt5 for the GUI and pyqtgraph for the image display.
"""

import sys
from PyQt5 import QtWidgets, QtCore, QtGui


class MainWindow(QtWidgets.QWidget):
    """
    The main window of the Lark tester.
    """

    def __init__(self, *args, **kwargs):
        """
        Sets up the main window
        """
        super().__init__(*args, **kwargs)

        # window size
        self.setMinimumSize(800, 600)
        self.setWindowTitle('Bax Analysis')


def exception_hook(type, value, traceback):
    """
    Use sys.__excepthook__, the standard hook.
    """
    sys.__excepthook__(type, value, traceback)


if __name__ == '__main__':

    # fix PyQt5 eating exceptions (see http://stackoverflow.com/q/14493081/1536976)
    sys.excepthook = exception_hook

    # create app
    app = QtWidgets.QApplication([])

    # show main window
    main_window = MainWindow()
    main_window.show()

    # start Qt app execution
    sys.exit(app.exec_())