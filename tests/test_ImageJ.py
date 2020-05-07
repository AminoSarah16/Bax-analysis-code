import imagej as ij
import os
import numpy as np
from PIL import Image


root_path = r'C:\Users\Sarah\Documents\Python\Bax-analysis\IF36_selected-for-analysis-with-Jan\results\mito-masks'
filename = 'IF36_spl15_U2OS-DKO_pcDNA-Bax-wt_6hEx_14hAct_cytC-AF488_Tom20-AF594_Bax-SR_cl9_superEarly-rings-on-one-side-only.tiff'

# tif image laden
file_path = os.path.join(root_path, filename)
img = np.array(Image.open(file_path))

# in imagej öffnen
ij.py.show(img, cmap='gray')
