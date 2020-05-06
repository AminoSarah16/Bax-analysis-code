"""
Finden der Segmentierungslinien im tif Bild (gelb: 255,255,0) und abspeichern als HDF5 File fomrat
Sarah probiert im Code herum
Warum ist es so langsam und kann man es schneller machen? FIJI plugin kann Hdf5 nicht lesen..
"""

import os
from PIL import Image  # https://pillow.readthedocs.io/en/stable/handbook/index.html
from utils.utils import *

separation_color = np.array([255, 255, 0])  # gelb
file_ending = "overlay_sgmt.tif"

if __name__ == '__main__':

    # Pfade
    root_path = get_root_path()
    print(root_path)
    mask_path = os.path.join(root_path, 'results', 'cell-masks')
    if not os.path.isdir(mask_path):
        os.makedirs(mask_path)

    for filename in os.listdir(root_path):  # ich erstelle eine Liste mit den Filenames in dem Ordner
        if filename.endswith(file_ending):  # wenn die Endung die Richtige ist, dann mach was damit, nämlich:
            print(filename)

            # tif image laden
            file_path = os.path.join(root_path, filename)
            img = np.array(Image.open(file_path))

            # binarize according to separation color
            # maske = img[:, :, 0] == 255 & img[:, :, 1] == 255 & img[:, :, 2] == 0
            maske = np.logical_not(np.logical_and.reduce((img[:, :, 0] == 255, img[:, :, 1] == 255, img[:, :, 2] == 0)))

            # check: display mask
            fig, ax = plt.subplots()
            #im = ax.imshow(maske, cmap='gray')
            #plt.show()

            # label (https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html)
            labelled_maske, number_cells = ndimage.measurements.label(maske)

            # check: display labelled mask
            im = ax.imshow(labelled_maske, cmap='inferno')
            plt.show()

            # save as tif (and save as hdf5)
            output_data = labelled_maske.astype(np.uint8)
            print(output_data.dtype)

            # save as tif
            img = Image.fromarray(output_data)
            output_file_path = os.path.join(mask_path, filename[:-len(file_ending)] + '.maske.tif')
            img.save(output_file_path, format='tiff') #schaut schwarz aus wegen LUT

            # save as hdf5
            #output_file_path = file_path[:-4] + '.maske.h5'
            #f = h5py.File(output_file_path, 'w')
            #dset = f.create_dataset("cell mask", data=output_data)
            #f.close()