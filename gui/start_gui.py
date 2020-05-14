"""
Starts the GUI, uses PyQt5 for the GUI and pyqtgraph for the image display.
"""

import os
import sys
import time
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import cv2
from matplotlib import cm
from PIL import Image  # https://pillow.readthedocs.io/en/stable/handbook/index.html
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
            _, tail = os.path.split(file)
            self.label.setText(tail)
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
        l.addWidget(self.img, stretch=1)
        ll = QtWidgets.QVBoxLayout()
        self.slider_rel_threshold = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_rel_threshold.setTickInterval(5)
        self.slider_rel_threshold.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider_rel_threshold.setSingleStep(1)
        self.slider_rel_threshold.setRange(-10, 10)
        self.slider_rel_threshold.valueChanged.connect(self.rel_threshold_changed)
        self.label_rel_threshold = QtWidgets.QLabel()
        ll.addWidget(self.label_rel_threshold)
        ll.addWidget(self.slider_rel_threshold)

        self.slider_blocksize = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_blocksize.setTickInterval(50)
        self.slider_blocksize.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider_blocksize.setSingleStep(10)
        self.slider_blocksize.setRange(100, 400)
        self.slider_blocksize.valueChanged.connect(self.blocksize_changed)
        self.label_blocksize = QtWidgets.QLabel()
        ll.addWidget(self.label_blocksize)
        ll.addWidget(self.slider_blocksize)

        self.button_toggle = QtWidgets.QPushButton('Show mask')
        self.button_toggle.setCheckable(True)
        self.button_toggle.toggled.connect(self.button_toggled)
        ll.addWidget(self.button_toggle)
        button_save = QtWidgets.QPushButton('Save')
        button_save.clicked.connect(self.save_mito_mask)
        ll.addWidget(button_save)
        ll.addStretch()
        l.addLayout(ll)

        # top layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(groupbox_file)
        layout.addWidget(groupbox_detection, stretch=1)

        # initialization
        self.denoised_data = None
        self.mito_mask = None
        self.slider_rel_threshold.setValue(-2)
        self.slider_blocksize.setValue(200)

    def rel_threshold_changed(self, value):
        self.label_rel_threshold.setText('Threshold ({})'.format(value))
        self.update_mask()
        self.update_image()

    def blocksize_changed(self, value):
        self.label_blocksize.setText('Blocksize ({})'.format(value))
        self.update_mask()
        self.update_image()

    def update_mask(self):
        if self.denoised_data is None:
            return
        self.mito_mask = cv2.adaptiveThreshold(self.denoised_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.slider_blocksize.value()*2+1, self.slider_rel_threshold.value())

    def update_image(self):
        if self.button_toggle.isChecked():
            if self.mito_mask is not None:
                self.img.getImageItem().setImage(self.mito_mask)
        else:
            if self.denoised_data is not None:
                self.img.getImageItem().setImage(self.denoised_data)

    def button_toggled(self, checked):
        if checked:
            self.button_toggle.setText('Show data')
        else:
            self.button_toggle.setText('Show mask')
        self.update_image()

    def save_mito_mask(self):
        """

        """
        if self.mito_mask is not None:

            # save everything
            output_path = os.path.join(mask_path, self.filename[:-4] + '.denoised.tiff')
            img = Image.fromarray(self.denoised_data)
            img.save(output_path, format='tiff')

            output_path = os.path.join(mask_path, self.filename[:-4] + '.tiff')
            img = Image.fromarray(self.denoised_data)
            img.save(output_path, format='tiff')

            self.status.emit('Mask saved.')


    def load_file(self, file):
        """

        """
        _, self.filename = os.path.split(file)
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


def detect_bax_cluster(denoised, pixel_sizes, cluster_threshold):
    '''
    Detektiert einzelne Cluster

    :param denoised:
    :param pixel_sizes:
    :return:
    '''

    # maske erstellen
    maske = denoised > cluster_threshold
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

def skeletonize_and_detect_holes(mask):
    # skeleton berechnen
    skel_label, skel_number = ndimage.measurements.label(mask, structure=structuring_element_block)

    # labeln der Löcher
    holes_label, holes_number = ndimage.measurements.label(skel_label == 0, structure=structuring_element_cross)

    # zählen der löcher pro skeleton
    skeleton_statistics = np.zeros((skel_number+1, 4))
    for i in range(1, skel_number + 1):
        m = skel_label == i
        number_pixel_skeleton = np.sum(m)
        m_left = np.roll(m, 1, axis=0)
        m_right = np.roll(m, -1, axis=0)
        m_up = np.roll(m, 1, axis=1)
        m_down = np.roll(m, -1, axis=1)
        a = holes_label[m_left]
        b = holes_label[m_right]
        c = holes_label[m_up]
        d = holes_label[m_down]
        e = np.concatenate((a,b,c,d))
        e = e[e > 1] # ignore 0 (pixel shifted is on another pixel of the skeleton) and 1 (big hole welches nicht zählt)

        # f enthält die Nummern der Löcher des Skeletons
        holes = np.unique(e)

        # Anzahl der Skeleton pixel an den Löchern
        number_pixel_around_hole = np.sum((a > 1) | (b > 1) | (c > 1) | (d > 1))
        pixel_around_holes_ratio = number_pixel_around_hole / number_pixel_skeleton

        # Anzahl der Pixel in allen Löchern
        number_pixel_in_holes = 0
        for j in range(len(holes)):
            number_pixel_in_holes += np.sum(holes_label == holes[j])

        skeleton_statistics[i, :] = [holes.size, pixel_around_holes_ratio, number_pixel_in_holes, number_pixel_skeleton]   # Anzahl der Löcher, Verhältnis Pixel des Skeletons an einem Loch zu Gesamtzahl, Gesamtfläche der Löcher in Pixeln


    return skel_label, skel_number, skeleton_statistics


def detect_bax_structures(denoised, pixel_sizes, segment_threshold, minimal_skeleton_size, dilation_size, progress):

    start = time.time()
    # segmentieren mit kleineren intensity threshold
    maske = denoised > segment_threshold

    #display_image(cmaske, 'kleine segment weg')

    # skeleton der maske
    skeleton = morphology.skeletonize(maske)

    # labeln des skeletons und bereinigen um kleine skeletons
    skel_label, skel_number = ndimage.measurements.label(skeleton == 1, structure=structuring_element_block)
    objects = ndimage.measurements.find_objects(skel_label)
    for i in range(skel_number):
        obj = objects[i]
        m = skel_label[obj[0], obj[1]]
        number_pixels = np.sum(m == i + 1)
        if number_pixels < minimal_skeleton_size: # skeleton must have at least a certain number of pixels
            m[m == i + 1] = 0
            skel_label[obj[0], obj[1]] = m
    # skel_label enthält alle Skeletons ab einer gewissen Größe und durchnummeriert

    progress.emit(3)
    # print('20 =  {}s'.format(time.time() - start))

    # Löcher detektieren und klassifizieren
    skel_label, skel_number, holes_statistics = skeletonize_and_detect_holes(skel_label != 0)

    progress.emit(35)
    # print('30 =  {}s'.format(time.time() - start))

    # copy lines to another array, um sie nochmal testen zu können
    lines = np.zeros(skel_label.shape)
    for i in range(1, skel_number + 1):
        m = skel_label == i
        if holes_statistics[i, 0] == 0:  # kein Loch -> Linie
            skel_label[m] = 0
            lines[m] = 1  # copy to lines mask

    progress.emit(41)
    # print('60 =  {}s'.format(time.time() - start))

    # dilation and skeletonize von Linien und dann nochmal klassifizieren, um nicht vollständig geschlossene Ringe zu schließen
    # lines = ndimage.gaussian_filter(lines, sigma=5)
    for i in range(dilation_size):
        lines = morphology.binary_dilation(lines, structuring_element_cross)
    lines = morphology.skeletonize(lines)
    # TODO falls es wesentlich weniger Pixel werden bei manchen Linien, dann hat der Skeletonalgorithmus sie zusammengezogen, das wollen wir eher nicht und sollten diese vielleicht wiederherstellen

    progress.emit(50)
    # print('70 =  {}s'.format(time.time() - start))

    lines_skel_label, _, _ = skeletonize_and_detect_holes(lines)

    progress.emit(66)
    # print('80 =  {}s'.format(time.time() - start))

    skel_label, skel_number, holes_statistics = skeletonize_and_detect_holes((skel_label > 0) | (lines_skel_label > 0))

    progress.emit(95)
    #print('90 =  {}s'.format(time.time() - start))

    # add one column in holes_statistics
    holes_statistics = np.append(holes_statistics, np.zeros((holes_statistics.shape[0], 1)), axis=1)
    classified_skeletons = np.zeros(skel_label.shape)
    for i in range(1, skel_number + 1):
        m = skel_label == i
        if holes_statistics[i, 0] == 0:  # kein Loch -> Linie
            typ = 1 # Linie
        elif holes_statistics[i, 0] == 1 and holes_statistics[i, 1] > 0.5: # ein Loch und nicht zuviel (50%) drum herum = Ring  # TODO: in die GUI
            typ = 2 # Ring
        else:  # komnplexe Struktur ist der gesamte rest
            typ = 3 # Komnplex
        classified_skeletons[m] = typ
        holes_statistics[i, 4] = typ

    progress.emit(100)
    #print('100 =  {}s'.format(time.time() - start))

    return classified_skeletons, skel_label, holes_statistics


class BaxPhenotypeWindow(QtWidgets.QWidget):
    """

    """

    #: signal, wants to send a status update
    status = QtCore.pyqtSignal(str)

    #: signal, wants to set progress bar
    progress = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        """
        Sets up the Mito detection.
        """
        super().__init__(*args, **kwargs)

        self.mode = 'cluster'  # 'cluster' or 'structures'

        # custom glasbey on dark colormap
        colormap = cm.get_cmap("hsv")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        np.random.shuffle(lut)
        lut = np.vstack(([0, 0, 0, 0], lut))
        # Apply the colormap
        # self.img.getImageItem().setLookupTable(lut)
        self.glasbey = pg.ColorMap(pos=np.linspace(0.0, 1.0, 20), color=lut.astype(np.uint8))

        self.colormap_grey = pg.ColorMap(pos = [0, 1], color = [(0, 0, 0, 255), (255, 255, 255, 255)])
        self.colormap_tricolor = pg.ColorMap(pos=[0, 0.33, 0.66, 1], color=[(0, 0, 0, 255), (255, 255, 0, 255), (255, 0, 255, 255), (0, 255, 255, 255)])

        # file group box
        groupbox_file = FileSelectionGroupBox()
        groupbox_file.selected.connect(self.load_file)

        # detection group box
        groupbox_detection = QtWidgets.QGroupBox('Phenotype')
        l = QtWidgets.QHBoxLayout(groupbox_detection)
        self.img = pg.ImageView()
        l.addWidget(self.img, stretch=1)

        ll = QtWidgets.QVBoxLayout()
        self.label_threshold = QtWidgets.QLabel()
        ll.addWidget(self.label_threshold)
        self.slider_threshold = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_threshold.setTickInterval(5)
        self.slider_threshold.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider_threshold.setSingleStep(1)
        self.slider_threshold.setRange(1, 50)
        self.slider_threshold.valueChanged.connect(self.threshold_changed)
        ll.addWidget(self.slider_threshold)

        self.label_structures_threshold = QtWidgets.QLabel()
        ll.addWidget(self.label_structures_threshold)
        self.slider_structures_threshold = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_structures_threshold.setTickInterval(1)
        self.slider_structures_threshold.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider_structures_threshold.setSingleStep(0.5)
        self.slider_structures_threshold.setRange(0, 10)
        self.slider_structures_threshold.valueChanged.connect(self.structures_threshold_changed)
        ll.addWidget(self.slider_structures_threshold)

        self.combobox_show = QtWidgets.QComboBox(self)
        self.combobox_show.addItems(['Data', 'Cluster', 'Structures', 'Structure types'])
        self.combobox_show.setCurrentIndex(0)
        self.combobox_show.currentIndexChanged.connect(self.show)
        ll.addWidget(self.combobox_show)

        button_save = QtWidgets.QPushButton('Save')
        button_save.clicked.connect(self.save)
        ll.addWidget(button_save)
        ll.addStretch()
        l.addLayout(ll)

        # top layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(groupbox_file)
        layout.addWidget(groupbox_detection, stretch=1)

        # initialization
        self.cluster_mask = None
        self.denoised_data = None
        self.slider_threshold.setValue(15)
        self.slider_structures_threshold.setValue(1.5)

    def threshold_changed(self, value):
        self.label_threshold.setText('Cluster threshold ({})'.format(value))
        self.update_cluster()
        self.update_image()

    def structures_threshold_changed(self, value):
        self.label_structures_threshold.setText('Structures threshold ({})'.format(value))
        self.update_structures()
        self.update_image()

    def update_cluster(self):
        if self.denoised_data is not None:
            self.cluster_mask = detect_bax_cluster(self.denoised_data, self.pixel_sizes, self.slider_threshold.value())

    def update_structures(self):
        if self.denoised_data is not None:
            # strukturen detektieren
            self.classified_skeletons, self.skel_label, self.holes_statistics = detect_bax_structures(self.denoised_data, self.pixel_sizes, self.slider_structures_threshold.value(), 20, 10, self.progress)

    def save(self):
        """

        """
        # save denoised
        output_path = os.path.join(bax_path, self.filename[:-4] + '.denoised.tiff')
        img = Image.fromarray(self.denoised_data)
        img.save(output_path, format='tiff')

        # save cluster maske
        output_path = os.path.join(bax_path, self.filename[:-4] + '.cluster.tiff')
        img = Image.fromarray(self.cluster_mask)
        img.save(output_path, format='tiff')

        self.status.emit('Everything is saved')

    def show(self, index):
        print(index)
        self.update_image()
        
    def update_image(self):
        if self.denoised_data is None:
            return
        idx = self.combobox_show.currentIndex()
        if idx == 0: # data
            self.img.getImageItem().setImage(self.denoised_data)
            # self.img.setImage(self.denoised_data)
            self.img.setColorMap(self.colormap_grey)
        elif idx == 1: # cluster
            self.img.getImageItem().setImage(self.cluster_mask)
            # self.img.setImage(self.cluster_mask)
            self.img.setColorMap(self.glasbey)
        elif idx == 2: # skel labels
            self.img.getImageItem().setImage(self.skel_label)
            self.img.setColorMap(self.glasbey)
        elif idx == 3: # skel type
            self.img.getImageItem().setImage(self.classified_skeletons)
            self.img.setColorMap(self.colormap_tricolor)

    def load_file(self, file):
        """

        """
        self.mode = 'cluster'
        _, self.filename = os.path.split(file)
        # bax stack laden
        bax_stacks = read_stack_from_imspector_measurement(file, 'STAR RED_STED')
        if len(bax_stacks) != 1:
            self.status.emit('No "STAR RED_STED" stack in measurement!')
            return
        stack = bax_stacks[0]
        image, self.pixel_sizes = extract_image_from_imspector_stack(stack)  # in den utils zu finden
        self.denoised_data = denoise_image(image)

        # cluster detektieren
        self.update_cluster()
        self.update_structures()
        self.update_image()

        self.status.emit('Loaded. Please adjust thresholds.')


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

        # progress bar
        self.progressbar = QtWidgets.QProgressBar()
        self.statusBar().addWidget(self.progressbar)
        self.progressbar.setVisible(False)

        # statusbar
        self.setStatus('Started')

    def start_mito_detection(self):
        widget = MitoDetectionWindow()
        widget.status.connect(self.setStatus)
        self.setCentralWidget(widget)

    def start_bax_phenotype(self):
        widget = BaxPhenotypeWindow()
        widget.status.connect(self.setStatus)
        widget.progress.connect(self.setProgressBar)
        self.setCentralWidget(widget)

    def setStatus(self, message):
        self.statusBar().showMessage(message, 2000)

    def setProgressBar(self, value):
        if value == 100:
            self.progressbar.setVisible(False)
            return
        if not self.progressbar.isVisible():
            self.progressbar.setVisible(True)
        self.progressbar.setValue(value)



def exception_hook(type, value, traceback):
    """
    Use sys.__excepthook__, the standard hook.
    """
    sys.__excepthook__(type, value, traceback)


if __name__ == '__main__':

    # fix PyQt5 eating exceptions (see http://stackoverflow.com/q/14493081/1536976)
    sys.excepthook = exception_hook

    # paths
    root_path = get_root_path()
    gui_path = os.path.dirname(__file__)
    mask_path = os.path.join(root_path, 'results', 'mito-masks')
    if not os.path.isdir(mask_path):
        os.makedirs(mask_path)
    bax_path = os.path.join(root_path, 'results', 'bax-structures')
    if not os.path.isdir(bax_path):
        os.makedirs(bax_path)

    # create app
    app = QtWidgets.QApplication([])
    icon_file = os.path.join(gui_path, 'icons8-microscope-30.png')
    app.setWindowIcon(QtGui.QIcon(icon_file))

    # show main window
    main_window = MainWindow()
    main_window.show()

    # start Qt app execution
    sys.exit(app.exec_())