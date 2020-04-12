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

#tif image, wo ich die Linien gezogen habe laden
img_w_borders = np.array(Image.open(file_path))

# binarize according to separation color
separation_color = np.array([255, 255, 0])
#print(separation_color)
#print(separation_color.shape)
dimensions = img_w_borders.shape #gibt die dimensionen des arrays des Border-Bildes zurück und scheint ein tuple zu sein
mask = np.ones(dimensions[0:2]) #Creates an array aus lauter 1en mit den ersten beiden Dimensionen von dims [0:2] bedeute bis
# exkl 2, also position 0 und 1. (Ich könnte auch np.ones(img_w_borders.shape[0:2]) schreiben)
#print(mask)
for x in range(dimensions[0]): #Für die x dimension gehe durch die Werte des Attributes 0 von dem array der Dimensionen
    for y in range(dimensions[1]): #Für die y dimension gehe durch die Werte des Attributes 1
        # is it yellow at (x,y)?
        if all(img_w_borders[x, y, :] == separation_color): #Wenn der Wert an xy an der dritten Stelle : genau der
            # Separation color entspricht, dann gehe in die if Funktion und führe das aus
            #numpy-function "all" tests whether all array elements along a given axis evaluate to True
            #SYNTAX??? was macht der Doppelpunkt?

            # ja, also lösche aus der maske, bzw gib ihm an der Stelle den Wert 0 statt 1.
            mask[x, y] = 0

# check: display mask
fig, ax = plt.subplots() # Creates just a figure and only one subplot
# #plt. zu schreiben reicht hier, da das ganze Paket matplotlib.pyplot oben als plt imported is
im = ax.imshow(mask, cmap='gray') #=?
plt.show() #display all figures and block until the figures have been closed
#A single experimental keyword argument, block, may be set to True or False
# to override the blocking behavior described above.

# label (https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html)
labeled_mask, number_cells = ndimage.measurements.label(mask) #Label features in an array.
#Any non-zero values in input are counted as features and zero values are considered the background.
# Da die Linien ja 0 sind, und die Zellen 1, werden die als Objekte nummeriert, getrennt von den 0-Linien.
# check: display labeled mask
#im = ax.imshow(labeled_maske, cmap='inferno')
#plt.show()

# save as hdf5

output_file_path = file_path[:-4] + '.maske.h5' #we remove the last 4 characters, which is ".tif"
# and add the string instead
f = h5py.File(output_file_path, 'w') #w stands for write, because we want to write the file to the harddrive
dset = f.create_dataset("cell mask", data=labeled_mask) #H5py Datasets are very similar to NumPy arrays. They are
# homogeneous collections of data elements, with an immutable datatype and (hyper)rectangular shape
f.close()