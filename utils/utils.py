"""
Hilfsfunktionen für alle Skripte.
"""

import specpy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import configparser
import skimage.morphology as morphology

structuring_element_block = ndimage.morphology.generate_binary_structure(2, 2)  # 3x3 block
structuring_element_cross = ndimage.morphology.generate_binary_structure(2, 1)  # 3x3 cross


def get_root_path():
    """
    Retrieves the root path
    """
    config = configparser.ConfigParser()
    config.read('../bax-analysis.ini')
    return config['general']['root-path']

def read_stack_from_imspector_measurement(file_path, name_part):
    """
    Lädt Stacks aus einer Imspector Messung.

    :param file_path: Pfad der Imspector Messung.
    :param name_part: Teil des Stacknamens
    :return: Alles Stacks, die so heißen
    """

    # File lesen
    im_file = sp.File(file_path, sp.File.Read)
    number_stacks = im_file.number_of_stacks()

    # lese alle stacks in eine liste
    stacks = []
    for i in range(number_stacks):
        stack = im_file.read(i)
        stacks.append(stack)

    # finde allen stacks, deren name name_part enthält
    stacks = [stack for stack in stacks if name_part in stack.name()]

    return stacks


def extract_image_from_imspector_stack(stack):
    """
    
    :param stack: Imspector stack 
    :return: Numpy array
    """
    data = stack.data()

    # Dimensionnen sind [1,1,Ny,Nx] wir wollen aber [Nx, Ny]

    # reduce to [Ny, Nx]
    size = data.shape
    data = np.reshape(data, size[2:])

    # transponieren [Nx, Ny]
    data = np.transpose(data)

    lengths = stack.lengths()
    pixel_sizes = (lengths[0] / data.shape[0] / 1e-6, lengths[1] / data.shape[1] / 1e-6)  # conversion m to µm

    return data, pixel_sizes

# our colormaps
green_on_black_colormap = mpl.colors.LinearSegmentedColormap.from_list('custom',
                                             [(0,    'black'),
                                              (1,    'Green')], N=255)
magenta_on_black_colormap = mpl.colors.LinearSegmentedColormap.from_list('custom',
                                             [(0,    'black'),
                                              (1,    'Magenta')], N=255)


def display_image(image, title, cmaps=('rainbow',)*10):
    """
    kann jetzt ein oder mehrere Bilder gleichzeitig darstellen
    :param image: image to display
    """

    # falls image kein tuple ist, mache ein tuple der größe 1 daraus
    if not isinstance(image, tuple):
        image = (image,)   # (..,) ist die syntax für ein tuple der größe 1
        title = (title,)

    n = len(image)
    fig, axes = plt.subplots(1, n)
    if not isinstance(axes, np.ndarray):
        axes = (axes,) # muss auch für axes gemacht werden
    for i in range(n):
        axes[i].imshow(image[i], cmap=cmaps[i])
        axes[i].set_title(title[i])
    plt.show()


def scale_to_255(image):
    return image / np.amax(image) * 2550


def denoise_image(image, sigma=2):

    # Rauschen reduzieren
    denoised = ndimage.gaussian_filter(image, sigma=2)

    return denoised