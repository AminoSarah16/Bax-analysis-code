from utils.utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



root_path = get_root_path()
print(root_path)
file = os.path.join(root_path, 'IF36_spl15_U2OS-DKO_pcDNA-Bax-wt_6hEx_14hAct_cytC-AF488_Tom20-AF594_Bax-SR_cl8_ringheaven_STED-overlay.tif')
print(file)
file.plt.show()

# https://gist.github.com/jgomezdans/402500
# display_image(file, ('test'))
# A random colormap for matplotlib
# Glasbey = matplotlib.colors.ListedColormap ( numpy.random.rand ( 256,3))
# print(Glasbey)


# https://gist.github.com/jgomezdans/402500 >> answer
# vals = np.linspace(0,1,256)
# np.random.shuffle(vals)
# cmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))

# TODO: wieso geht das nur mit dieser labeled mask?
vals = np.linspace(0, 1, 256)
np.random.shuffle(vals)
cmap = plt.cm.colors.ListedColormap(plt.cm.rainbow(vals))

# check: display labelled mask
im = ax.imshow(labelled_maske, cmap=cmap)
plt.show()