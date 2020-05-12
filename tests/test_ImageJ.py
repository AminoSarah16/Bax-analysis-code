import imagej
ij = imagej.init(r'C:\Users\Sarah\Downloads\fiji-win64\Fiji.app')
# ij.getVersion()
import os
import numpy as np
from PIL import Image
from utils.utils import *


root_path = r'C:\Users\Sarah\Documents\Python\Bax-analysis\IF36_selected-for-analysis-with-Jan\results\bax-structures'
filename = 'IF36_spl15_U2OS-DKO_pcDNA-Bax-wt_6hEx_14hAct_cytC-AF488_Tom20-AF594_Bax-SR_cl8_ringheaven.structures-id.tiff'

# tif image laden
file_path = os.path.join(root_path, filename)
img = np.array(Image.open(file_path))
display_image(img, 'title')  # wieso kann ich jetzt eigentlich generell keine cmap mehr auswählen?

#https://nbviewer.jupyter.org/github/imagej/tutorials/blob/master/notebooks/1-Using-ImageJ/6-ImageJ-with-Python-Kernel.ipynb
# in imagej öffnen
# ij.py.show(img, cmap = 'gray')




image = ij.io().open(img)


