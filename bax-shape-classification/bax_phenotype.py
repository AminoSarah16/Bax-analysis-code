"""
Loads BAX images and preprocesses them (denoise, remove background) and then detects clusters, rings, ...
"""

import os
import specpy as sp
import numpy as np
from PIL import Image  # https://pillow.readthedocs.io/en/stable/handbook/index.html
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import skimage.filters as filters
import skimage.morphology as morphology
import cv2
from utils.utils import *

s1 = ndimage.morphology.generate_binary_structure(2, 2)  # 3x3 block
s2 = ndimage.morphology.generate_binary_structure(2, 1)  # 3x3 cross

def detect_cluster(denoised, pixel_sizes):
    '''
    Detektiert einzelne Cluster

    :param denoised:
    :param pixel_sizes:
    :return:
    '''

    # maske erstellen
    maske = denoised > 15  # TODO anpassen auf die anderen Bilder

    # labeln der cluster-segment und größe pro segment
    labeled_mask, number_clusters = ndimage.measurements.label(maske)

    for i in range(1, number_clusters + 1):
        cluster_pixelsum = np.sum(labeled_mask == i)
        cluster_area = cluster_pixelsum * pixel_sizes[0] * pixel_sizes[1]
        # print('cluster {} contains {} pixels, area = {} µm²'.format(i, cluster_pixelsum, cluster_area))

    # display_image(maske, 'bax cluster')

    return labeled_mask

def skeletonize_and_detect_holes(mask):
    # skeleton berechnen
    skel_label, skel_number = ndimage.measurements.label(mask, structure=s1)

    # labeln der Löcher
    holes_label, holes_number = ndimage.measurements.label(skel_label == 0, structure=s2)

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
        f = np.unique(e)

        # Anzahl der Skeleton pixel an den Löchern
        number_pixel_around_hole = np.sum((a > 1) | (b > 1) | (c > 1) | (d > 1))
        pixel_around_holes_ratio = number_pixel_around_hole / number_pixel_skeleton

        # Anzahl der Pixel in allen Löchern
        number_pixel_in_holes = 0
        for j in range(len(f)):
            number_pixel_in_holes += np.sum(holes_label == f[j])

        skeleton_statistics[i, :] = [f.size, pixel_around_holes_ratio, number_pixel_in_holes, number_pixel_skeleton]   # Anzahl der Löcher, Verhältnis Pixel des Skeletons an einem Loch zu Gesamtzahl, Gesamtfläche der Löcher in Pixeln


    return skel_label, skel_number, skeleton_statistics


def detect_structures(denoised, pixel_sizes):

    # segmentieren mit kleineren threshold
    maske = denoised > 1.5

    #display_image(cmaske, 'kleine segment weg')

    # skeleton der maske
    skeleton = morphology.skeletonize(maske)

    # labeln des skeletons und bereinigen um kleine skeletons
    skel_label, skel_number = ndimage.measurements.label(skeleton == 1, structure=s1)
    for i in range(1, skel_number + 1):
        m = skel_label == i
        number_pixels = np.sum(m)
        if number_pixels < 20:  # skeleton must have at least 20 pixel
            skel_label[m] = 0
    # skel_label enthält alle Skeletons ab einer gewissen Größe und durchnummeriert

    # Löcher detektieren und klassifizieren
    skel_label, skel_number, holes_statistics = skeletonize_and_detect_holes(skel_label != 0)

    # copy lines to another array
    lines = np.zeros(skel_label.shape)
    for i in range(1, skel_number + 1):
        m = skel_label == i
        if holes_statistics[i, 0] == 0:  # kein Loch -> Linie
            skel_label[m] = 0
            lines[m] = 1  # copy to lines mask

    # dilation and skeletonize von Linien und dann nochmal klassifizieren, um nicht vollständig geschlossene Ringe zu schließen
    # lines = ndimage.gaussian_filter(lines, sigma=5)
    for i in range(10):
        lines = morphology.binary_dilation(lines, s2)
    lines = morphology.skeletonize(lines)

    lines_skel_label, _, _ = skeletonize_and_detect_holes(lines)

    skel_label, skel_number, holes_statistics = skeletonize_and_detect_holes((skel_label > 0) | (lines_skel_label > 0))

    # add one column in holes_statistics
    holes_statistics = np.append(holes_statistics, np.zeros((holes_statistics.shape[0], 1)), axis=1)
    classified_skeletons = np.zeros(skel_label.shape)
    for i in range(1, skel_number + 1):
        m = skel_label == i
        if holes_statistics[i, 0] == 0:  # kein Loch -> Linie
            typ = 1
        elif holes_statistics[i, 0] == 1 and holes_statistics[i, 1] > 0.5: # ein Loch und nicht zuviel drum herum
            typ = 2
        else:  # komnplexe Struktur ist der gesamte rest
            typ = 3
        classified_skeletons[m] = typ
        holes_statistics[i, 4] = typ

    display_image((classified_skeletons, skel_label), ('type of skeletons', 'labelled skeletons'))

    np.set_printoptions(precision=3, suppress=True)
    for i in range(1, holes_statistics.shape[0]):
        print('id {}: {}'.format(i, holes_statistics[i, :]))


def detect_bax_structures(file_path):

    # bax stack laden
    bax_stacks = read_stack_from_imspector_measurement(file_path, 'STAR RED_STED')

    if len(bax_stacks) != 1:
        print('Problem: {} bax stacks, need one.'.format(len(bax_stacks)))
        return
    stack = bax_stacks[0]

    # Daten des bax stacks und Rauschen entfernen
    image, pixel_sizes = extract_image_from_imspector_stack(stack)  # in den utils zu finden
    denoised = denoise_image(image)

    # cluster detektieren
    cluster_mask = detect_cluster(denoised, pixel_sizes)
    display_image(cluster_mask, 'cluster')

    # display
    # display_image((scale_to_255(image), scale_to_255(denoised), cluster_mask), ('original', 'denoised', 'cluster'), (green_on_black_colormap, green_on_black_colormap, 'rainbow'))

    # strukturen detektieren
    structures = detect_structures(denoised, pixel_sizes)




if __name__ == '__main__':

    root_path = r'C:\Users\Sarah\Documents\Python\Bax-analysis\IF36_selected-for-analysis-with-Jan'

    for filename in os.listdir(root_path):  # ich erstelle eine Liste mit den Filenames in dem Ordner
        if filename.endswith(".msr"):  # wenn die Endung .msr ist, dann mach was damit, nämlich:
            print(filename)

            file_path = os.path.join(root_path, filename)
            output = detect_bax_structures(file_path)

            break  # for testing purposes only the first measurement