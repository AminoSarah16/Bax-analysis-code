import os
from PIL import Image  # https://pillow.readthedocs.io/en/stable/handbook/index.html
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import h5py  # https://docs.h5py.org/en/stable/

'''
Finden der Segmentierungslinien im tif Bild (gelb: 255,255,0) und abspeichern als HDF5 File fomrat
Sarah probiert im Code herum
Warum ist es so langsam und kann man es schneller machen? FIJI plugin kann Hdf5 nicht lesen..
'''

# Pfade
root_path = r'Q:\00_Users\Sarah Schweighofer (sschwei)\Freiburg\IF36_selected-for-analysis-with-Jan'
file_name = r'IF36_spl21_U2OS-DKO_pcDNA-Bax-H5i_6hEx_14hAct_cytC-AF488_Tom20-AF594_Bax-SR_cl6-7_dotty-andbigdots_STED-overlay_sgmt.tif'
file_path = os.path.join(root_path, file_name)

# tif image laden
img = np.array(Image.open(file_path))

# binarize according to separation color
separation_color = np.array([255, 255, 0])
# maske = img[:, :, 0] == 255 & img[:, :, 1] == 255 & img[:, :, 2] == 0
maske = np.logical_not(np.logical_and.reduce((img[:, :, 0] == 255, img[:, :, 1] == 255, img[:, :, 2] == 0)))


# check: display mask
fig, ax = plt.subplots()
im = ax.imshow(maske, cmap='gray')
plt.show()

# label (https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html)
labelled_maske, number_cells = ndimage.measurements.label(maske)

# check: display labelled mask
im = ax.imshow(labelled_maske, cmap='inferno')
plt.show()

# save as tif and save as hdf5
output_data = labelled_maske.astype(np.uint8)
print(output_data.dtype)

# save as tif
img = Image.fromarray(output_data)
output_file_path = file_path[:-4] + '.maske.tiff'
img.save(output_file_path, format = 'tiff') #schaut schwarz aus wegen LUT

# save as hdf5
output_file_path = file_path[:-4] + '.maske.h5'
f = h5py.File(output_file_path, 'w')
dset = f.create_dataset("cell mask", data=output_data)
f.close()