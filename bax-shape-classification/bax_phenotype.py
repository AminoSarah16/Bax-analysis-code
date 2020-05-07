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
import time
from utils.utils import *

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


def detect_structures(denoised, pixel_sizes):

    # segmentieren mit kleineren threshold
    maske = denoised > 1.5

    #display_image(cmaske, 'kleine segment weg')

    # skeleton der maske
    skeleton = morphology.skeletonize(maske)

    # labeln des skeletons und bereinigen um kleine skeletons
    skel_label, skel_number = ndimage.measurements.label(skeleton == 1, structure=structuring_element_block)
    for i in range(1, skel_number + 1):
        m = skel_label == i
        number_pixels = np.sum(m)
        if number_pixels < 20:  # skeleton must have at least 20 pixel
            skel_label[m] = 0
    # skel_label enthält alle Skeletons ab einer gewissen Größe und durchnummeriert

    # Löcher detektieren und klassifizieren
    skel_label, skel_number, holes_statistics = skeletonize_and_detect_holes(skel_label != 0)

    # copy lines to another array, um sie nochmal testen zu können
    lines = np.zeros(skel_label.shape)
    for i in range(1, skel_number + 1):
        m = skel_label == i
        if holes_statistics[i, 0] == 0:  # kein Loch -> Linie
            skel_label[m] = 0
            lines[m] = 1  # copy to lines mask

    # dilation and skeletonize von Linien und dann nochmal klassifizieren, um nicht vollständig geschlossene Ringe zu schließen
    # lines = ndimage.gaussian_filter(lines, sigma=5)
    for i in range(10):
        lines = morphology.binary_dilation(lines, structuring_element_cross)
    lines = morphology.skeletonize(lines)
    # TODO falls es wesentlich weniger Pixel werden bei manchen Linien, dann hat der Skeletonalgorithmus sie zusammengezogen, das wollen wir eher nicht und sollten diese vielleicht wiederherstellen

    lines_skel_label, _, _ = skeletonize_and_detect_holes(lines)

    skel_label, skel_number, holes_statistics = skeletonize_and_detect_holes((skel_label > 0) | (lines_skel_label > 0))

    # add one column in holes_statistics
    holes_statistics = np.append(holes_statistics, np.zeros((holes_statistics.shape[0], 1)), axis=1)
    classified_skeletons = np.zeros(skel_label.shape)
    for i in range(1, skel_number + 1):
        m = skel_label == i
        if holes_statistics[i, 0] == 0:  # kein Loch -> Linie
            typ = 1 # Linie
        elif holes_statistics[i, 0] == 1 and holes_statistics[i, 1] > 0.5: # ein Loch und nicht zuviel drum herum = Ring
            typ = 2 # Ring
        else:  # komnplexe Struktur ist der gesamte rest
            typ = 3 # Komnplex
        classified_skeletons[m] = typ
        holes_statistics[i, 4] = typ

    display_image((skeleton, skel_label > 0), ('erster skeleton', 'final'))

    display_image((classified_skeletons, skel_label), ('type of skeletons', 'labelled skeletons'))

    np.set_printoptions(precision=3, suppress=True)
    for i in range(1, holes_statistics.shape[0]):
        print('id {}: {}'.format(i, holes_statistics[i, :]))

    return classified_skeletons, skel_label, holes_statistics


def detect_bax_structures(root_path, filename, bax_path):
    """
    Detect Bax Strukturen in einer Messung (Cluster und andere)
    """

    file_path = os.path.join(root_path, filename)

    # bax stack laden
    bax_stacks = read_stack_from_imspector_measurement(file_path, 'STAR RED_STED')

    if len(bax_stacks) != 1:
        print('Problem: {} bax stacks, need one.'.format(len(bax_stacks)))
        return
    stack = bax_stacks[0]

    # Daten des bax stacks und Rauschen entfernen
    image, pixel_sizes = extract_image_from_imspector_stack(stack)  # in den utils zu finden
    denoised = denoise_image(image)

    # save structures type (1,2,3) (classified_skeletons)
    output_path = os.path.join(bax_path, filename[:-4] + '.denoised.tiff')
    img = Image.fromarray(denoised)
    img.save(output_path, format='tiff')

    # cluster detektieren
    cluster_mask = detect_cluster(denoised, pixel_sizes)
    display_image(cluster_mask, 'cluster')

    # save cluster maske
    output_path = os.path.join(bax_path, filename[:-4] + '.cluster.tiff')
    img = Image.fromarray(cluster_mask)
    img.save(output_path, format='tiff')

    # display
    # display_image((scale_to_255(image), scale_to_255(denoised), cluster_mask), ('original', 'denoised', 'cluster'), (green_on_black_colormap, green_on_black_colormap, 'rainbow'))

    # strukturen detektieren
    classified_skeletons, skel_label, holes_statistics = detect_structures(denoised, pixel_sizes)

    # store detected structures

    # save structures type (1,2,3) (classified_skeletons)
    output_path = os.path.join(bax_path, filename[:-4] + '.structures-type.tiff')
    img = Image.fromarray(classified_skeletons)
    img.save(output_path, format='tiff')

    # save structures id (skel_label)
    output_path = os.path.join(bax_path, filename[:-4] + '.structures-id.tiff')
    img = Image.fromarray(skel_label)
    img.save(output_path, format='tiff')

    # save structures statistics (holes_statistics)
    output_path = os.path.join(bax_path, filename[:-4] + '.structures.csv')
    np.savetxt(output_path, holes_statistics, delimiter=',', fmt='%f')


if __name__ == '__main__':

    root_path = get_root_path()
    bax_path = os.path.join(root_path, 'results', 'bax-structures')
    if not os.path.isdir(bax_path):
        os.makedirs(bax_path)

    filenames = list(os.listdir(root_path))
    filenames.reverse()
    for filename in filenames:  # ich erstelle eine Liste mit den Filenames in dem Ordner
        if filename.endswith(".msr"):  # wenn die Endung .msr ist, dann mach was damit, nämlich:
            print(filename)

            detect_bax_structures(root_path, filename, bax_path)

            # break  # for testing purposes only the first measurement