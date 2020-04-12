"""
  Laden einiger Imspector Messungen, Anzeige, ...
"""

import os
import specpy as sp
import numpy as np
import matplotlib.pyplot as plt

# alle Einstellungen
root_path = r'Q:\00_Users\Sarah Schweighofer (sschwei)\Freiburg\IF36_selected-for-analysis-with-Jan'
file_path = os.path.join(root_path, 'IF36_spl15_U2OS-DKO_pcDNA-Bax-wt_6hEx_14hAct_cytC-AF488_Tom20-AF594_Bax-SR_cl8_ringheaven.msr')

# File lesen
im_file = sp.File(file_path, sp.File.Read)
number_stacks = im_file.number_of_stacks()
print('Messung {} hat {} Bilder.'.format(file_path, number_stacks))

# alle Stacks lesen
# if False:
for i in range(number_stacks):
    stack = im_file.read(i)
    # print('Stack {} ist {}D mit der Größe {}'.format(stack.name(), stack.number_of_dimensions(), stack.sizes()))

    # ist es der STED BAX Kanal?
    if stack.name().startswith('STAR RED_STED'):
        # Ja
        data = stack.data()

        # Dimensionnen sind [1,1,Ny,Nx] wir wollen aber [Nx, Ny]

        # reduce to [Ny, Nx]
        size = data.shape
        data = np.reshape(data, size[2:])

        # transponieren [Nx, Ny]
        data = np.transpose(data)

        # anzeigen
        #fig, ax = plt.subplots()
        #im = ax.imshow(data, cmap='Greens')
        #plt.show()

fig, ax = plt.subplots()
im = ax.imshow(np.zeros((100, 100)), cmap='Greens')
plt.show()

del im_file
print('finished')
