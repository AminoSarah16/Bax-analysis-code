"""
  Macht eine Maske von einem Mito Channel image
"""

import os
import specpy as sp
import numpy as np
from PIL import Image  # https://pillow.readthedocs.io/en/stable/handbook/index.html
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import skimage.filters as filters
import cv2
from utils.utils import *

def create_mito_mask_from_measurement(root_path, filename):
    """
    Liest Imspector file und erstellt Maske
    """
    filepath = os.path.join(root_path, filename)
    mito_stacks = read_stack_from_imspector_measurement(filepath, 'Alexa 594_STED')
    #für ein Element in der Liste stack, wenn dieses Element Alex 594 im Namen hat, dann schreibe das Element da rein

    if len(mito_stacks) != 1:
        print('Problem: {} mito stacks, need one.'.format(len(mito_stacks)))
        return

    mito_stack = mito_stacks[0]
    data = mito_stack.data()

    # Dimensionnen sind [1,1,Ny,Nx] wir wollen aber [Nx, Ny]

    # reduce to [Ny, Nx]
    size = data.shape
    data = np.reshape(data, size[2:])

    # transponieren [Nx, Ny]
    data = np.transpose(data)

    # display
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(filename)
    ax1.imshow(data, cmap='Greens')
    ax1.set_title('original mito image')
    # plt.show()

    # Rauschen reduzieren
    denoised_data = ndimage.gaussian_filter(data, sigma=2)

    # display
    ax2.imshow(denoised_data, cmap='Greens')
    ax2.set_title('denoised mito image')
    plt.show()

    # super threshold
    denoised_data = denoised_data.astype(np.uint8)  # ich wandle mein denoised data mit astype in 8bit image um

    maske = cv2.adaptiveThreshold(denoised_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 401, -2)
    # So funktioniert der opencv adaptive threshold: neuesBild = cv.adaptiveThreshold(source, maxValue, adaptiv

    return denoised_data, maske

if __name__ == '__main__':
    # haupt code untendrunter

    # iteriere über alle Messungen

    # alle Einstellungen
    root_path = r'Q:\00_Users\Sarah Schweighofer (sschwei)\Freiburg\IF36_selected-for-analysis-with-Jan'
    mask_path = os.path.join(root_path, 'Masks')
    if not os.path.isdir(mask_path):
        os.makedirs(mask_path)

    for filename in os.listdir(root_path):  # ich erstelle eine Liste mit den Filenames in dem Ordner
        if filename.endswith(".msr"):  # wenn die Endung .msr ist, dann mach was damit, nämlich:
            print(filename)

            output = create_mito_mask_from_measurement(root_path, filename)

            if output is None:
                # could not create mask, continue
                continue

            denoised_data = output[0]
            maske = output[1]

            # save everything
            output_path = os.path.join(mask_path, filename[:-4] + '.denoised.tiff')
            img = Image.fromarray(denoised_data)
            img.save(output_path, format='tiff')

            output_path = os.path.join(mask_path, filename[:-4] + '.maske.tiff')
            img = Image.fromarray(maske)
            img.save(output_path, format='tiff')

