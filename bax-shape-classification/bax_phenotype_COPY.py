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
import cv2
from utils.utils_COPY import *


def detect_cluster(denoised, pixel_sizes): #is used within the load-bax function below
    # maske erstellen
    maske = denoised > 15  # TODO anpassen auf die anderen Bilder

    # labeln der cluster-segment und größe pro segment
    labeled_mask, number_clusters = ndimage.measurements.label(maske) #.label function has been used in single_cell_
    #separation > check there

    for i in range(1, number_clusters + 1):
        cluster_pixelsum = np.sum(labeled_mask == i)
        cluster_area = cluster_pixelsum * pixel_sizes[0] * pixel_sizes[1]
        print('cluster {} contains {} pixels, area = {} µm²'.format(i, cluster_pixelsum, cluster_area))

    display_image(maske, 'bax cluster')

    return labeled_mask


def detect_structures(denoised, pixel_sizes, cluster_mask): #is used within the load-bax function below

    # segmentieren mit kleinerem threshold
    maske = denoised > 2

    # ignore all structures with area < a certain number of pixels
    labeled_mask, number_segments = ndimage.measurements.label(maske)

    for i in range(1, number_segments + 1): #durchsuche alle abgeschlossenen elemente in der Maske
        #(die ganzen Nullen sind die Bereiche, wo nix is, daher muss er bei 1 anfangen und bis number_segments+1 gehen)
        segment_pixelsum = np.sum(labeled_mask == i) #summiere die Anzahl der Pixel pro Element (i) der Maske
        if segment_pixelsum < 200: #wenn die Fläche kleiner ist
            maske[labeled_mask == i] = 0 #dann mache diese Flächen auf 0 (= verschwinden aus der Maske: an der Position
            #wo er diese Flächensumme < 200 gefunden hat wird eine Null ins Array geschrieben)

    # ignore all structures with overlap with cluster mask > X%
    labeled_mask, number_segments = ndimage.measurements.label(maske)

    for i in range(1, number_segments + 1):
        segment_pixelsum = np.sum(labeled_mask == i)
        cluster_in_segment_pixelsum = np.sum((labeled_mask == i) & (cluster_mask != 0)) #Syntax mit Klammern wichtig
        #summiere die Anzahl der Pixel von jenen Elementen die in der Structures-Maske enthalten sind und in der Cluster
        #-Maske nicht null sind (also jene cluster, die auch in Strukturen enthalten sind)
        if cluster_in_segment_pixelsum / segment_pixelsum > 0.3: #wenn der Überlapp von Cluster mit anderer Struktur
        #mehr als 30% beträgt, wenn also die Struktur zu mehr als 30% aus Clustern besteht

            # ist nur ein cluster, ignorieren
            maske[labeled_mask == i] = 0 #, dann nimm den Cluster aus der Strukturen-Maske raus.

    display_image((maske, cluster_mask > 0), ('bax ringe', 'bax cluster')) #cluster_mask > 0 macht nur, dass die Cluster
    # alle in einer Farbe angezeigt werden, weil sie ja in der Maske mit 1 bis unendlich gelabelt wurden und daher
    # veschieden gut sichtbar sind normalerweise. Hiermit sind sie alle gleich eingefärbt, weil das wieder so eine Art
    # threshold ist



def load_bax_image(file_path): #is used within the main code

    # bax stack laden
    bax_stacks = read_stack_from_imspector_measurement(file_path, 'STAR RED_STED')

    if len(bax_stacks) != 1:
        print('Problem: {} bax stacks, need one.'.format(len(bax_stacks)))
        return
    stack = bax_stacks[0]

    image, pixel_sizes = extract_image_from_imspector_stack(stack)

    display_image(image, 'bax original')

    denoised = denoise_image(image)

    # cluster detektieren
    cluster_mask = detect_cluster(denoised, pixel_sizes)

    # strukturen detektieren
    structures = detect_structures(denoised, pixel_sizes, cluster_mask)




if __name__ == '__main__':

    root_path = r'Q:\00_Users\Sarah Schweighofer (sschwei)\Freiburg\IF36_selected-for-analysis-with-Jan'

    for filename in os.listdir(root_path):  # ich erstelle eine Liste mit den Filenames in dem Ordner
        if filename.endswith(".msr"):  # wenn die Endung .msr ist, dann mach was damit, nämlich:
            print(filename)

            file_path = os.path.join(root_path, filename)
            output = load_bax_image(file_path) #here the whole function gets used, which includes the
            #above functions

            break  # for testing purposes only the first measurement