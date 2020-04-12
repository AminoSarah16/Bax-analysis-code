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

def save_as_tif(output_path, data):
    img = Image.fromarray(data.astype(np.uint8))
    img.save(output_path, format='tiff')

# alle Einstellungen
root_path = r'Q:\00_Users\Sarah Schweighofer (sschwei)\Freiburg\IF36_selected-for-analysis-with-Jan'
file_path = os.path.join(root_path, 'IF36_spl15_U2OS-DKO_pcDNA-Bax-wt_6hEx_14hAct_cytC-AF488_Tom20-AF594_Bax-SR_cl8_ringheaven.msr')

# File lesen
im_file = sp.File(file_path, sp.File.Read)
number_stacks = im_file.number_of_stacks()
print('Messung {} hat {} Bilder.'.format(file_path, number_stacks))

# nach dem mito Kanal suchen in allen Stacks
found_mito_channel = False
for i in range(number_stacks):
    stack = im_file.read(i)
    # print('Stack {} ist {}D mit der Größe {}'.format(stack.name(), stack.number_of_dimensions(), stack.sizes()))

    # ist es der Mito Kanal?
    if stack.name().startswith('Alexa 594_STED'):
        # Ja
        found_mito_channel = True
        data = stack.data()

        # Dimensionnen sind [1,1,Ny,Nx] wir wollen aber [Nx, Ny]

        # reduce to [Ny, Nx]
        size = data.shape
        data = np.reshape(data, size[2:])

        # transponieren [Nx, Ny]
        data = np.transpose(data)

        break
if not found_mito_channel:
    raise RuntimeError('Kein Mito Kanal in dieser Messung.')

# display
fig, (ax1, ax2) = plt.subplots(1, 2)
im = ax1.imshow(data, cmap='Greens')
ax1.set_title('original mito image')
plt.show()

# Rauschen reduzieren
denoised_data = ndimage.gaussian_filter(data, sigma=2)

# display
im = ax2.imshow(denoised_data, cmap='Greens')
ax2.set_title('denoised mito image')
plt.show()

# adaptiver theshold für maske
#fig, ax = filters.try_all_threshold(denoised_data)
#plt.show()
maske_isodata = denoised_data > filters.threshold_isodata(denoised_data) #True vs false
maske_otsu = denoised_data > filters.threshold_otsu(denoised_data)

# save everything
output_path = file_path[:-4] + '.denoised.tiff'
save_as_tif(output_path, denoised_data)
output_path = file_path[:-4] + '.threshold-isodata.tiff'
save_as_tif(output_path, maske_isodata)
output_path = file_path[:-4] + '.threshold-otsu.tiff'
save_as_tif(output_path, maske_otsu)


