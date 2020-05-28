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


def read_text(file):
    """
    Reads a whole text file (UTF-8 encoded).
    """
    with open(file, mode='r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    return text


def write_text(file, text):
    """
    Writes a whole text file (UTF-8 encoded).
    """
    with open(file, mode='w', encoding='utf-8') as f:
        f.write(text)

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


def read_sted_stacks_from_imspector_measurement(file_path):
    """
    Mit Hilfe von Sarah und aufgrund der vielen Probleme mit den Stacks hier die komplizierte Formel um die MitoStacks
    herauszuholen. Eigentlich heißen die Stacks "Alexa 594_STED" aber halt nicht immer...
    """

    # File lesen
    im_file = sp.File(file_path, sp.File.Read)
    number_stacks = im_file.number_of_stacks()

    # lese alle stacks in eine liste
    stacks = []
    for i in range(number_stacks):
        stack = im_file.read(i)
        stacks.append(stack)

    sted_stacks = [stack for stack in stacks if " " not in stack.name() or "STED" in stack.name() or "Ch2 {2}" in stack.name() or "Ch4 {2}" in stack.name()]

    # if we get more than 2 stacks (one AF594 and one STAR RED) then it's most likely duplicates and we will just remove them from the list
    if len(sted_stacks) > 2:
        sted_stacks = sted_stacks[:2]

    return sted_stacks


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

    # data as float32
    data = data.astype(np.float32)

    # compute pixel sizes
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
    TODO: besserer Kommentar, damit sich Sarah auch auskennt
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


def clean_image(image, noise_sigma=2, background_sigma=20, subtraction_fraction=0.8):
    """
    Smooths with a small Gaussian to reduce noise and with a large Gaussian to remove background, then subtracts part
    of the background.
    """

    if image.dtype != np.float32:
        raise RuntimeError('Need floating point precision for input.')

    # Rauschen reduzieren
    denoised_image = ndimage.gaussian_filter(image, sigma=noise_sigma)

    # Und Hintergrund abziehen
    background_image = ndimage.gaussian_filter(image, sigma=background_sigma)

    clean_image = np.maximum(0, denoised_image - subtraction_fraction * background_image)

    # print('{}, {}, {}, {}'.format(np.sum(image), np.sum(denoised), np.sum(background), np.sum(filtered)))

    return clean_image, denoised_image, background_image