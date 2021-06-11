"""
Starts the GUI, uses PyQt5 for the GUI and pyqtgraph for the image display.
"""

import os
import sys
import time
import json
from functools import partial
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import cv2
from matplotlib import cm
from PIL import Image  # https://pillow.readthedocs.io/en/stable/handbook/index.html
from utils.utils import *

# TODO Löcher in Clustern schließen in Cluster Detektion
# TODO spektral crosstalk
# TODO Mindestgröße für Mitos


def saveColorMapState(histogramLUTItem):
    return {
        'gradient': histogramLUTItem.gradient.saveState(),
        'levels': histogramLUTItem.getLevels()
        #'mode': histogramLUTItem.levelMode,
    }


def restoreColorMapState(histogramLUTItem, state):
    #histogramLUTItem.setLevelMode(state['mode'])
    histogramLUTItem.gradient.restoreState(state['gradient'])
    histogramLUTItem.setLevels(*state['levels'])


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

        self.current_path = root_path

    def select_file(self):
        """

        """
        file = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', self.current_path, 'Measurement (*.msr);;All files (*.*)')
        file = file[0]
        if not file or not os.path.isfile(file):
            self.label.setText('')
        else:
            dirpath, filename = os.path.split(file)
            self.current_path = dirpath
            self.label.setText(filename)
            self.selected.emit(file)

def create_spinbox(minimum, maximum, step, value_changed_slot):
    """

    """
    spinbox = QtWidgets.QSpinBox()
    spinbox.setRange(minimum, maximum)
    spinbox.setSingleStep(step)
    spinbox.valueChanged.connect(value_changed_slot)
    return spinbox


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

        # selection of image to show
        self.combobox_show = QtWidgets.QComboBox(self)
        image_displays = ['Raw Data', 'Clean data', 'Mito mask']
        self.combobox_show.addItems(image_displays)
        self.combobox_show.currentIndexChanged.connect(self.update_image)
        ll.addWidget(self.combobox_show)

        # signal fwhm
        ll.addWidget(QtWidgets.QLabel('Signal FWHM (nm)'))
        self.spinbox_signal_fwhm = create_spinbox(40, 500, 20, self.schedule_update)
        ll.addWidget(self.spinbox_signal_fwhm)

        # background fwhm
        ll.addWidget(QtWidgets.QLabel('Background FWHM (nm)'))
        self.spinbox_background_fwhm = create_spinbox(500, 5000, 100, self.schedule_update)
        ll.addWidget(self.spinbox_background_fwhm)

        # subtraction factor
        ll.addWidget(QtWidgets.QLabel('Background subtraction (%)'))
        self.spinbox_bg_subtraction_factor = create_spinbox(0, 100, 5, self.schedule_update)
        ll.addWidget(self.spinbox_bg_subtraction_factor)

        ## threshold factor
        ll.addWidget(QtWidgets.QLabel('Rel. threshold (%)'))
        self.spinbox_rel_threshold = create_spinbox(0, 100, 1, self.schedule_update)
        ll.addWidget(self.spinbox_rel_threshold)

        # save button
        button_save = QtWidgets.QPushButton('Save')
        button_save.clicked.connect(self.save_mito_mask)
        ll.addStretch()
        ll.addWidget(button_save)
        l.addLayout(ll)

        # top layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(groupbox_file)
        layout.addWidget(groupbox_detection, stretch=1)

        # timer
        self.timer_update = QtCore.QTimer()
        self.timer_update.setSingleShot(True)
        self.timer_update.setInterval(500)
        self.timer_update.timeout.connect(self.update)

        # shortcuts
        shortcut_raw = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_1), self)
        shortcut_raw.activated.connect(partial(self.combobox_show.setCurrentIndex, 0))
        shortcut_clean = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_2), self)
        shortcut_clean.activated.connect(partial(self.combobox_show.setCurrentIndex, 1))
        shortcut_mito = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_3), self)
        shortcut_mito.activated.connect(partial(self.combobox_show.setCurrentIndex, 2))

        # initialization
        self.raw_data = None
        self.clean_data = None
        self.mito_mask = None
        self.current_index = None
        self.spinbox_signal_fwhm.setValue(200)
        self.spinbox_background_fwhm.setValue(2000)
        self.spinbox_bg_subtraction_factor.setValue(50)
        self.spinbox_rel_threshold.setValue(10)
        self.colorhistogram_item_states = [None] * len(image_displays)
        self.combobox_show.setCurrentIndex(0)

    def schedule_update(self):
        self.timer_update.start()

    def update(self):
        """
        Compute everything new and update the display
        """
        if self.raw_data is None:
            return

        # status
        self.status.emit('Updating')

        # denoise image
        px = self.pixel_sizes[0] * 1000 # convert to nm
        noise_sigma = self.spinbox_signal_fwhm.value() / px / 2.35  # conversion FWHM to sigma = 2.35
        background_sigma = self.spinbox_background_fwhm.value() / px / 2.35
        subtraction_fraction = self.spinbox_bg_subtraction_factor.value() / 100
        self.clean_data, self.denoised_data, self.background_data = clean_image(self.raw_data, noise_sigma, background_sigma, subtraction_fraction)
        # self.denoised_data = self.denoised_data.astype(np.uint8)  # ich wandle mein denoised data mit astype in 8bit image um

        # create mask
        self.update_mask()

        # update image
        self.update_image()

    def update_mask(self):
        rel_threshold = self.spinbox_rel_threshold.value() / 100
        self.mito_mask = self.clean_data > np.max(self.clean_data) * rel_threshold
        self.mito_mask = self.mito_mask.astype(np.uint8)
        # self.mito_mask = cv2.adaptiveThreshold(self.denoised_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.slider_blocksize.value()*2+1, self.slider_rel_threshold.value())

    def update_image(self, _=None):
        if self.raw_data is None:
            return
        index = self.combobox_show.currentIndex()
        # store state
        if self.current_index is not None:
            self.colorhistogram_item_states[self.current_index] = saveColorMapState(self.img.getHistogramWidget().item)
        if index == 0: # raw data
            self.img.getImageItem().setImage(self.raw_data)
            self.img.setColorMap(colormap_hot)
        elif index == 1: # clean data
            self.img.getImageItem().setImage(self.clean_data)
            self.img.setColorMap(colormap_hot)
        elif index == 2: # mito mask
            self.img.getImageItem().setImage(self.mito_mask)
            self.img.setColorMap(colormap_grey)
        if self.colorhistogram_item_states[index] is not None:
            restoreColorMapState(self.img.getHistogramWidget().item, self.colorhistogram_item_states[index])
        self.current_index = index

    def save_mito_mask(self):
        """

        """
        if self.mito_mask is not None:

            # save mito mask
            output_path = os.path.join(mask_path, self.filename[:-4] + '.mito-mask.tiff')
            if os.path.isfile(output_path):
                # already existing, ask before overwriting
                answer = QtWidgets.QMessageBox.warning(self, 'File exists', 'Mito mask already saved, overwrite?', QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
                if answer == QtWidgets.QMessageBox.Cancel:
                    return
            img = Image.fromarray(self.mito_mask)
            img.save(output_path, format='tiff')

            # save clean
            output_path = os.path.join(mask_path, self.filename[:-4] + '.clean.tiff')
            img = Image.fromarray(self.clean_data)
            img.save(output_path, format='tiff')

            # save parameters
            parameters = {
                'file': self.filename,
                'signal-fwhm': self.spinbox_signal_fwhm.value(),
                'background-fwhm': self.spinbox_background_fwhm.value(),
                'background-subtraction-factor': self.spinbox_bg_subtraction_factor.value(),
                'relative-threshold': self.spinbox_rel_threshold.value()
            }
            # output
            output_path = os.path.join(mask_path, self.filename[:-4] + '.mito-mask-parameters.json')
            text = json.dumps(parameters, indent=1)
            write_text(output_path, text)

            self.status.emit('Mask saved.')


    def load_file(self, file):
        """

        """
        _, self.filename = os.path.split(file)
        # load mito image
        sted_stacks = read_sted_stacks_from_imspector_measurement(file)
        # mito_stacks = read_stack_from_imspector_measurement(file, 'Alexa 594_STED')
        mito_stack = sted_stacks[0]  # mito stack is the first sted stack??
        self.raw_data, self.pixel_sizes = extract_image_from_imspector_stack(mito_stack)

        # update
        self.schedule_update()

        # status message
        self.status.emit('Loaded. Please adjust parameters.')

        # TODO if file already exist, maybe load old parameters and set them
        # test for existing output
        output_path = os.path.join(mask_path, self.filename[:-4] + '.mito-mask.tiff')
        if os.path.isfile(output_path):
            self.status.emit('Mito mask file already existing. Save will overwrite.')


def detect_bax_cluster(denoised, pixel_sizes, cluster_threshold, minimal_area_size):
    '''
    Detektiert einzelne Cluster

    :param denoised:
    :param pixel_sizes:
    :return:
    '''

    # maske erstellen
    maske = denoised > cluster_threshold
    minimal_area = 3.14 * (minimal_area_size)**2 # cluster müssen mindestens eine fläche von 100nm2PI haben

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
    # print('das dauerte jetzt {}s'.format(time.time() - start))

    labeled_mask, number_clusters = ndimage.measurements.label(labeled_mask > 0)

    # display_image((maske, labeled_mask), ('bax cluster', 'area filtered'))

    return labeled_mask, number_clusters


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


def detect_bax_structures(denoised, segment_threshold, minimal_skeleton_size, dilation_size, progress):

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

        # file group box
        groupbox_file = FileSelectionGroupBox()
        groupbox_file.selected.connect(self.load_file)

        # detection group box
        groupbox_detection = QtWidgets.QGroupBox('Phenotype')
        l = QtWidgets.QHBoxLayout(groupbox_detection)
        self.img = pg.ImageView()
        l.addWidget(self.img, stretch=1)

        ll = QtWidgets.QVBoxLayout()

        # selection of image to show
        self.combobox_show = QtWidgets.QComboBox(self)
        image_displays = ['Raw Data', 'Clean Data', 'Cluster', 'Structures', 'Structure types']
        self.combobox_show.addItems(image_displays)
        self.combobox_show.currentIndexChanged.connect(self.update_image)
        ll.addWidget(self.combobox_show)

        # signal fwhm
        ll.addWidget(QtWidgets.QLabel('Signal FWHM (nm)'))
        self.spinbox_signal_fwhm = create_spinbox(20, 500, 20, self.schedule_update)
        ll.addWidget(self.spinbox_signal_fwhm)

        # background fwhm
        ll.addWidget(QtWidgets.QLabel('Background FWHM (nm)'))
        self.spinbox_background_fwhm = create_spinbox(500, 5000, 100, self.schedule_update)
        ll.addWidget(self.spinbox_background_fwhm)

        # subtraction factor
        ll.addWidget(QtWidgets.QLabel('Background subtraction (%)'))
        self.spinbox_bg_subtraction_factor = create_spinbox(0, 100, 5, self.schedule_update)
        ll.addWidget(self.spinbox_bg_subtraction_factor)

        self.label_clusters = QtWidgets.QLabel('Det. Clusters: 0')
        ll.addWidget(self.label_clusters)

        # cluster threshold
        ll.addWidget(QtWidgets.QLabel('Cluster: rel. threshold (%)'))
        self.spinbox_cluster_rel_threshold = create_spinbox(0, 100, 1, self.schedule_update)
        ll.addWidget(self.spinbox_cluster_rel_threshold)

        # cluster minimal area
        ll.addWidget(QtWidgets.QLabel('Cluster: minimal area size (nm)'))
        self.spinbox_cluster_min_area_size = create_spinbox(0, 1000, 20, self.schedule_update)
        ll.addWidget(self.spinbox_cluster_min_area_size)

        # structures threshold
        ll.addWidget(QtWidgets.QLabel('Structures: rel. threshold (%)'))
        self.spinbox_structures_rel_threshold = create_spinbox(0, 100, 1, self.schedule_update)
        ll.addWidget(self.spinbox_structures_rel_threshold)

        # structures minimal skeleton length
        ll.addWidget(QtWidgets.QLabel('Structures: min. skeleton length'))
        self.spinbox_structures_min_skel_length = create_spinbox(0, 1000, 1, self.schedule_update)
        ll.addWidget(self.spinbox_structures_min_skel_length)

        # structures dilation strength
        ll.addWidget(QtWidgets.QLabel('Structures: dilation strength'))
        self.spinbox_structures_dilation_strength = create_spinbox(0, 100, 1, self.schedule_update)
        ll.addWidget(self.spinbox_structures_dilation_strength)

        # save button
        button_save = QtWidgets.QPushButton('Save')
        button_save.clicked.connect(self.save_bax_structures)
        ll.addStretch()
        ll.addWidget(button_save)
        l.addLayout(ll)

        # top layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(groupbox_file)
        layout.addWidget(groupbox_detection, stretch=1)

        # timer
        self.timer_update = QtCore.QTimer()
        self.timer_update.setSingleShot(True)
        self.timer_update.setInterval(500)
        self.timer_update.timeout.connect(self.update)

        # shortcuts
        shortcut_raw = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_1), self)
        shortcut_raw.activated.connect(partial(self.combobox_show.setCurrentIndex, 0))
        shortcut_clean = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_2), self)
        shortcut_clean.activated.connect(partial(self.combobox_show.setCurrentIndex, 1))
        shortcut_cluster = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_3), self)
        shortcut_cluster.activated.connect(partial(self.combobox_show.setCurrentIndex, 2))
        shortcut_structures = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_4), self)
        shortcut_structures.activated.connect(partial(self.combobox_show.setCurrentIndex, 3))
        shortcut_structures_type = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_5), self)
        shortcut_structures_type.activated.connect(partial(self.combobox_show.setCurrentIndex, 4))

        # initialization
        self.raw_data = None
        self.cluster_mask = None
        self.clean_data = None
        self.image_index_history = []

        self.spinbox_signal_fwhm.setValue(100)
        self.spinbox_background_fwhm.setValue(600)
        self.spinbox_bg_subtraction_factor.setValue(80)
        self.spinbox_cluster_rel_threshold.setValue(5)
        self.spinbox_cluster_min_area_size.setValue(10)
        self.spinbox_structures_rel_threshold.setValue(5)
        self.spinbox_structures_min_skel_length.setValue(20)
        self.spinbox_structures_dilation_strength.setValue(10)

        self.colorhistogram_item_states = [None] * len(image_displays)
        self.combobox_show.setCurrentIndex(0)

    def schedule_update(self):
        self.timer_update.start()

    def update(self):
        """
        Compute everything new and update display
        """
        if self.raw_data is None:
            return

        # status
        self.status.emit('Updating')
        print('updating')

        # denoise image
        px = self.pixel_sizes[0] * 1000 # convert to nm
        noise_sigma = self.spinbox_signal_fwhm.value() / px / 2.35  # conversion FWHM to sigma = 2.35
        background_sigma = self.spinbox_background_fwhm.value() / px / 2.35
        subtraction_fraction = self.spinbox_bg_subtraction_factor.value() / 100
        self.clean_data, self.denoised_data, self.background_data = clean_image(self.raw_data, noise_sigma, background_sigma, subtraction_fraction)

        self.update_cluster()
        self.update_structures()
        self.update_image()

    def update_cluster(self):
        if self.clean_data is not None:
            threshold = np.max(self.clean_data) * self.spinbox_cluster_rel_threshold.value() / 100
            minimal_area_size = self.spinbox_cluster_min_area_size.value() / 1000 # conversion to pixel_sizes
            self.cluster_mask, number_clusters = detect_bax_cluster(self.clean_data, self.pixel_sizes, threshold, minimal_area_size)
            self.label_clusters.setText('Det. Clusters: {}'.format(number_clusters))

    def update_structures(self):
        if self.clean_data is not None:
            # strukturen detektieren
            threshold = np.max(self.clean_data) * self.spinbox_structures_rel_threshold.value() / 100
            minimal_skeleton_size = self.spinbox_structures_min_skel_length.value()
            dilation_size = self.spinbox_structures_dilation_strength.value()
            # TODO temporarily out of order (concentrating on clusters)
            # self.classified_skeletons, self.skel_label, self.holes_statistics = detect_bax_structures(self.clean_data, threshold, minimal_skeleton_size, dilation_size, self.progress)
            self.classified_skeletons = self.clean_data
            self.skel_label = self.clean_data

    def save_bax_structures(self):
        """

        """
        # save cluster mask
        output_path = os.path.join(bax_path, self.filename[:-4] + '.cluster.tiff')
        # TODO check that tiff really stores more than 255 values
        if os.path.isfile(output_path):
            # already existing, ask before overwriting
            answer = QtWidgets.QMessageBox.warning(self, 'File exists', 'Cluster mask already saved, overwrite?',
                                                   QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            if answer == QtWidgets.QMessageBox.Cancel:
                return
        img = Image.fromarray(self.cluster_mask) # contains clusters as cluster mask
        img.save(output_path, format='tiff')

        # save denoised
        output_path = os.path.join(bax_path, self.filename[:-4] + '.denoised.tiff')
        img = Image.fromarray(self.clean_data)
        img.save(output_path, format='tiff')

        # save parameters
        parameters = {
            'file': self.filename,
            'pixel-sizes': self.pixel_sizes,
            'signal-fwhm': self.spinbox_signal_fwhm.value(),
            'background-fwhm': self.spinbox_background_fwhm.value(),
            'background-subtraction-factor': self.spinbox_bg_subtraction_factor.value(),
            'cluster-relative-threshold': self.spinbox_cluster_rel_threshold.value(),
            'cluster-min-area-size': self.spinbox_cluster_min_area_size.value(),
            'structures-min_skel_length': self.spinbox_structures_min_skel_length.value(),
            'structures-dilation-strength': self.spinbox_structures_dilation_strength.value()
        }
        # output
        output_path = os.path.join(bax_path, self.filename[:-4] + '.bax-parameters.json')
        text = json.dumps(parameters, indent=1)
        write_text(output_path, text)

        self.status.emit('Everything is saved')

    def update_image(self, _=None):
        if self.raw_data is None:
            return
        index = self.combobox_show.currentIndex()
        # store state
        if self.image_index_history:
            self.colorhistogram_item_states[self.image_index_history[-1]] = saveColorMapState(self.img.getHistogramWidget().item)
        if index == 0: # data
            self.img.getImageItem().setImage(self.raw_data)
            self.img.setColorMap(colormap_hot)
        elif index == 1: # clean data
            self.img.getImageItem().setImage(self.clean_data)
            self.img.setColorMap(colormap_hot)
        elif index == 2: # cluster
            self.img.getImageItem().setImage(self.cluster_mask)
            self.img.setColorMap(colormap_glasbey)
        elif index == 3: # skel labels
            self.img.getImageItem().setImage(self.skel_label)
            self.img.setColorMap(colormap_glasbey)
        elif index == 4: # skel type
            self.img.getImageItem().setImage(self.classified_skeletons)
            self.img.setColorMap(colormap_tricolor)
        if self.colorhistogram_item_states[index] is not None:
            restoreColorMapState(self.img.getHistogramWidget().item, self.colorhistogram_item_states[index])
        self.image_index_history.append(index)
        if len(self.image_index_history) > 2:
            self.image_index_history = self.image_index_history[-2:] # only keep last two

    def eventFilter(self, obj: 'QObject', event: 'QEvent') -> bool:
        if obj == self and event.type() == QtCore.QEvent.Wheel and event.button() == QtCore.Qt.MiddleButton and len(self.image_index_history) >= 2:
            self.combobox_show.setCurrentIndex(self.image_index_history[-2])
            event.accept()
            return True
        else:
            return super().eventFilter(obj, event)

    def load_file(self, file):
        """

        """
        self.mode = 'cluster'
        _, self.filename = os.path.split(file)
        # bax stack laden
        # bax_stacks = read_stack_from_imspector_measurement(file, 'STAR RED_STED')
        sted_stacks = read_sted_stacks_from_imspector_measurement(file)
        bax_stack = sted_stacks[1]  # mito stack is the second sted stack??
        self.raw_data, self.pixel_sizes = extract_image_from_imspector_stack(bax_stack)

        # TODO if results already exists, load parameters and apply

        # update
        self.schedule_update()

        # status message
        self.status.emit('Loaded. Please adjust parameters.')


class MainWindow(QtWidgets.QMainWindow):
    """
    The main window of the Bax analysis GUI.
    """

    def __init__(self, qapp):
        """
        Sets up the main window
        """
        super().__init__()

        self.qapp = qapp

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
        self.qapp.installEventFilter(widget)
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

    # colormaps

    # custom glasbey on dark colormap
    colormap = cm.get_cmap("hsv")  # cm.get_cmap("CMRmap")
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
    # np.random.shuffle(lut)
    lut = np.vstack(([0, 0, 0, 0], lut[0::10, :]))
    # Apply the colormap
    # self.img.getImageItem().setLookupTable(lut)
    colormap_glasbey = pg.ColorMap(pos=np.linspace(0.0, 1.0, 27), color=lut.astype(np.uint8))
    colormap_grey = pg.ColorMap(pos=[0, 1], color=[(0, 0, 0, 255), (255, 255, 255, 255)])
    colormap_flame = pg.ColorMap(pos = [0.0, 0.2, 0.5, 0.8, 1.0], color = [(0, 0, 0, 255), (7, 0, 220, 255), (236, 0, 134, 255), (246, 246, 0, 255), (255, 255, 255, 255)])
    colormap_hot = pg.ColorMap(pos=np.linspace(0.0, 1.0, 4),
                                    color=[(0, 0, 0, 255), (185, 0, 0, 255), (255, 220, 0, 255),
                                           (255, 255, 255, 255)])
    colormap_tricolor = pg.ColorMap(pos=[0, 0.33, 0.66, 1],
                                         color=[(0, 0, 0, 255), (255, 255, 0, 255), (255, 0, 255, 255),
                                                (0, 255, 255, 255)])

    # create app
    app = QtWidgets.QApplication([])
    # icon_file = os.path.join(gui_path, 'icons8-microscope-30.png')
    icon_file = os.path.join(gui_path, 'IF2_spl1_icon.png')
    app.setWindowIcon(QtGui.QIcon(icon_file))

    # show main window
    main_window = MainWindow(app)
    main_window.show()

    # start Qt app execution
    sys.exit(app.exec_())