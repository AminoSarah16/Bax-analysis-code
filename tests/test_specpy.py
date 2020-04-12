import os
import specpy as sp
import numpy as np
import matplotlib.pyplot as plt

root_path = r'Q:\00_Users\Sarah Schweighofer (sschwei)\Freiburg\IF36_selected-for-analysis-with-Jan'
file_path = os.path.join(root_path, 'IF36_spl15_U2OS-DKO_pcDNA-Bax-wt_6hEx_14hAct_cytC-AF488_Tom20-AF594_Bax-SR_cl8_ringheaven.msr')

# File lesen
im_file = sp.File(file_path, sp.File.Read)
number_stacks = im_file.number_of_stacks()
print('Messung {} hat {} Bilder.'.format(file_path, number_stacks))

# File schließen
del im_file

# File lesen
im_file = sp.File(file_path, sp.File.Read)
number_stacks = im_file.number_of_stacks()
print('Messung {} hat {} Bilder.'.format(file_path, number_stacks))

# File schließen
del im_file

fig, ax = plt.subplots()
im = ax.imshow(np.zeros((100, 100)))
plt.show()