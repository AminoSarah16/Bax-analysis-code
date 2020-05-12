"""
Starts the GUI, uses PyQt5 for the GUI and pyqtgraph for the image display.
"""

import os
import sys
import time
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import cv2
from utils.utils import *


def create_action(text, parent, destination):
    """

    """
    action = QtWidgets.QAction(text, parent)
    action.triggered.connect(destination)
    return action


class FileSelectionGroupBox(QtWidgets.QGroupBox):
    """

    """

    #: signal, new file has been selected
    selected = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        """
        Sets up the file selection group box.
        """
        super().__init__('File', *args, **kwargs)

        self.label = QtWidgets.QLineEdit('')
        self.label.setReadOnly(True)
        button = QtWidgets.QPushButton('...')
        button.clicked.connect(self.select_file)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(button)

    def select_file(self):
        """

        """
        file = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', root_path, 'Measurement (*.msr);;All files (*.*)')
        file = file[0]
        if not file or not os.path.isfile(file):
            self.label.setText('')
        else:
            self.label.setText(file)
            self.selected.emit(file)


class MitoDetectionWindow(QtWidgets.QWidget):
    """

    """

    #: signal, wants to send a status update
    status = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        """
        Sets up the Mito detection.
        """
        super().__init__(*args, **kwargs)

        # file group box
        groupbox_file = FileSelectionGroupBox()
        groupbox_file.selected.connect(self.load_file)

        # detection group box
        groupbox_detection = QtWidgets.QGroupBox('Detection')
        l = QtWidgets.QHBoxLayout(groupbox_detection)
        self.img = pg.ImageView()
        #self.win = pg.GraphicsLayoutWidget()
        #self.img = pg.ImageItem()
        #p1 = self.win.addPlot()
        #p1.addItem(self.img)
        #hist = pg.HistogramLUTItem()
        #hist.setImageItem(self.img)
        #self.win.addItem(hist)
        #l.addWidget(self.win)
        l.addWidget(self.img, stretch=1)
        ll = QtWidgets.QVBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setTickInterval(5)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider.setSingleStep(1)
        self.slider.setRange(-10, 10)
        self.slider.valueChanged.connect(self.threshold_changed)
        self.label_threshold = QtWidgets.QLabel()
        ll.addWidget(self.label_threshold)
        ll.addWidget(self.slider)
        self.button_toggle = QtWidgets.QPushButton('Show mask')
        self.button_toggle.setCheckable(True)
        self.button_toggle.toggled.connect(self.button_toggled)
        ll.addWidget(self.button_toggle)
        button_save = QtWidgets.QPushButton('Save')
        ll.addWidget(button_save)
        ll.addStretch()
        l.addLayout(ll)

        # top layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(groupbox_file)
        layout.addWidget(groupbox_detection, stretch=1)

        # initialization
        self.denoised_data = None
        self.mask = None
        self.slider.setValue(-2)

    def threshold_changed(self, value):
        self.label_threshold.setText('Threshold ({})'.format(value))
        self.update_mask()
        self.update_image()

    def update_mask(self):
        if self.denoised_data is None:
            return
        self.mask = cv2.adaptiveThreshold(self.denoised_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 401, self.slider.value())

    def update_image(self):
        if self.button_toggle.isChecked():
            if self.mask is not None:
                self.img.setImage(self.mask)
        else:
            if self.denoised_data is not None:
                self.img.setImage(self.denoised_data)

    def button_toggled(self, checked):
        if checked:
            self.button_toggle.setText('Show data')
        else:
            self.button_toggle.setText('Show mask')
        self.update_image()


    def load_file(self, file):
        """

        """
        # load mito image
        mito_stacks = read_stack_from_imspector_measurement(file, 'Alexa 594_STED')
        if len(mito_stacks) != 1:
            self.status.emit('No "Alexa 594_STED" stack in measurement!')
            return
        mito_stack = mito_stacks[0]
        data, pixel_sizes = extract_image_from_imspector_stack(mito_stack)

        # denoise image
        self.denoised_data = ndimage.gaussian_filter(data, sigma=2)
        self.denoised_data = self.denoised_data.astype(np.uint8)  # ich wandle mein denoised data mit astype in 8bit image um

        # create mask
        self.update_mask()
        self.update_image()

        self.status.emit('Loaded. Please adjust threshold.')


def detect_cluster(denoised, pixel_sizes):
    '''
    Detektiert einzelne Cluster

    :param denoised:
    :param pixel_sizes:
    :return:
    '''

    # maske erstellen
    maske = denoised > 15  # TODO anpassen auf die anderen Bilder
    minimal_area = 3.14 * (0.1)**2 # cluster müssen mindestens eine fläche von 100nm2PI haben

    # labeln der cluster-segment und größe pro segment
    labeled_mask, number_clusters = ndimage.measurements.label(maske)
    objects = ndimage.measurements.find_objects(labeled_mask)

    # filter by cluster area
    start = time.time()
    for i in range(number_clusters):
        obj = objects[i]
        m = labeled_mask[obj[0], obj[1]]  # das Rechteck, welches den Cluster i+1 enthält, ausschneiden
        cluster_pixelsum = np.sum(m == i + 1)  # Anzahl der Pixel in diesem Rechteck, die wen Wert i+1 haben, zählen
        cluster_area = cluster_pixelsum * pixel_sizes[0] * pixel_sizes[1]
        if cluster_area < minimal_area:
            m[m == i + 1] = 0  # alle Pixel, in dem Recheck, die den Wert i+1 haben, löschen
            labeled_mask[obj[0], obj[1]] = m  # das Rechteck in das gelabelten Bild wieder einsetzen
        # print('cluster {} contains {} pixels, area = {} µm²'.format(i, cluster_pixelsum, cluster_area))
    print('das dauerte jetzt {}s'.format(time.time() - start))

    labeled_mask, number_clusters = ndimage.measurements.label(labeled_mask > 0)

    # display_image((maske, labeled_mask), ('bax cluster', 'area filtered'))

    return labeled_mask


class BaxPhenotypeWindow(QtWidgets.QWidget):
    """

    """

    #: signal, wants to send a status update
    status = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        """
        Sets up the Mito detection.
        """
        super().__init__(*args, **kwargs)

        # file group box
        groupbox_file = FileSelectionGroupBox()
        groupbox_file.selected.connect(self.load_file)

        # detection group box
        groupbox_detection = QtWidgets.QGroupBox('Phenotype')
        l = QtWidgets.QHBoxLayout(groupbox_detection)
        self.img = pg.ImageView()
        l.addWidget(self.img, stretch=1)

        ll = QtWidgets.QVBoxLayout()
        self.button_cluster_toggle = QtWidgets.QPushButton('Show cluster')
        self.button_cluster_toggle.setCheckable(True)
        self.button_cluster_toggle.toggled.connect(self.button_cluster_toggled)
        ll.addWidget(self.button_cluster_toggle)
        ll.addStretch()
        l.addLayout(ll)

        # top layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(groupbox_file)
        layout.addWidget(groupbox_detection, stretch=1)

        # initialization

    def button_cluster_toggled(self, checked):
        if checked:
            self.button_cluster_toggle.setText('Show data')
        else:
            self.button_cluster_toggle.setText('Show cluster')
        self.update_image
        
    def update_image(self):
        if self.button_cluster_toggle.isChecked():
            if self.cluster_mask is not None:
                self.img.setImage(self.cluster_mask)
        else:
            if self.denoised_data is not None:
                self.img.setImage(self.denoised_data)

    def load_file(self, file):
        """

        """
        # bax stack laden
        bax_stacks = read_stack_from_imspector_measurement(file, 'STAR RED_STED')
        if len(bax_stacks) != 1:
            self.status.emit('No "STAR RED_STED" stack in measurement!')
            return
        stack = bax_stacks[0]
        image, pixel_sizes = extract_image_from_imspector_stack(stack)  # in den utils zu finden
        self.denoised_data = denoise_image(image)

        # cluster detektieren
        self.cluster_mask = detect_cluster(self.denoised_data, pixel_sizes)
        self.update_image()

        self.status.emit('Loaded. Please adjust threshold.')


class MainWindow(QtWidgets.QMainWindow):
    """
    The main window of the Bax analysis GUI.
    """

    def __init__(self, *args, **kwargs):
        """
        Sets up the main window
        """
        super().__init__(*args, **kwargs)

        # window properties
        self.setMinimumSize(800, 600)
        self.setWindowTitle('Bax Analysis')

        # menus
        menu_analysis = QtWidgets.QMenu('Analysis', self)
        menu_analysis.addAction(create_action('Mito Detection', self, self.start_mito_detection))
        menu_analysis.addAction(create_action('Bax Phenotype', self, self.start_bax_phenotype))
        menu_analysis.addSeparator()
        menu_analysis.addAction(create_action('Exit', self, self.close))
        self.menuBar().addMenu(menu_analysis)

        # statusbar
        self.setStatus('Started')

    def start_mito_detection(self):
        widget = MitoDetectionWindow()
        widget.status.connect(self.setStatus)
        self.setCentralWidget(widget)

    def start_bax_phenotype(self):
        widget = BaxPhenotypeWindow()
        widget.status.connect(self.setStatus)
        self.setCentralWidget(widget)

    def setStatus(self, message):
        self.statusBar().showMessage(message, 2000)


def exception_hook(type, value, traceback):
    """
    Use sys.__excepthook__, the standard hook.
    """
    sys.__excepthook__(type, value, traceback)


if __name__ == '__main__':

    # fix PyQt5 eating exceptions (see http://stackoverflow.com/q/14493081/1536976)
    sys.excepthook = exception_hook

    # root path
    root_path = get_root_path()

    # create app
    app = QtWidgets.QApplication([])

    # show main window
    main_window = MainWindow()
    main_window.show()

    # start Qt app execution
    sys.exit(app.exec_())