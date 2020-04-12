"""
  Laden einiger Imspector Messungen, Anzeige, ...
  Sarah probiert herum
"""

import os
import specpy as sp
import numpy as np
import matplotlib.pyplot as plt

# alle Einstellungen
root_path = r'Q:\00_Users\Sarah Schweighofer (sschwei)\Freiburg\IF36_selected-for-analysis-with-Jan'
file_name = 'IF36_spl15_U2OS-DKO_pcDNA-Bax-wt_6hEx_14hAct_cytC-AF488_Tom20-AF594_Bax-SR_cl8_ringheaven.msr'
file_path = os.path.join(root_path, file_name)


# File lesen
im_file = sp.File(file_path, sp.File.Read) #sp. sagt ihm, dass die Funktion aus dem Package kommt, warum brauch ich das?
number_stacks = im_file.number_of_stacks() #hier darf ich das sp. nicht hinzufügen - warum?
print('Messung {} hat {} Bilder.'.format(file_name, number_stacks))
'''#zwei geschwungene Klammern brauchen zwei Variablen
#mit der .format Funktion kann man in nem String geschwungene Klammern als Platzhalter lassen und die Variablen danach
#einsetzen
# Das wäre die Deppen-Version: print("Messung " + str(file_name) + "hat " + str(number_stacks) + ".")
'''

# alle Stacks/Kanäle lesen
for i in range(number_stacks):
    stack = im_file.read(i)
    '''
    # für die Anzahl an Stacks (Bei Abberior ist ein Kanal automatisch immer ein Stack) geht
    # er durch und liest die einzelnen Kanäle ein und jedes dieser Elemente hat dann den Namen stack.
    # Diese .read Funktion ist aber was anderes als die .Read Funktion von SpecPy, richtig?
    # Nun könnte man sich noch ausgeben lassen, wie viele Dimensionen jeder Stack/Kanal hat
    # print('Stack {} ist {}D mit der Größe {}'.format(stack.name(), stack.number_of_dimensions(), stack.sizes()))
    # man könnte auch nur: print(stack.name(), stack.number_of_dimensions(), stack.sizes()
    '''

    '''# ist es der STED BAX Kanal? wir checken ob der Name des Kanals mit dem str beginnt und wenn ja schreiben wir die
    # Daten des Stacks als numpy array in die Variable channel_data
    '''
    if stack.name().startswith('STAR RED_STED'):
        '''# str.startswith(str, beg=0,end=len(string)). Python string method startswith() checks whether string starts
        # with str, optionally restricting the matching with the given indices start and end.
        '''
        channel_data = stack.data()
        #stack.data() returns the data of the stack as a NumPy array,
        #print(channel_data)

        # Dimensionnen sind [1,1,Ny,Nx] wir wollen aber [Nx, Ny] - wie haben wir das nochmal rausgefunden?
        # reduce to [Ny, Nx]
        channel_size = channel_data.shape
        print(channel_data.shape)
        # The shape attribute for numpy arrays returns the dimensions of the array. If Y
        # has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n
        #print(channel_size)
        channel_data2 = np.reshape(channel_data, channel_size[2:])
        #reshape spits out the part of the data you want to have. Here we only want the last two infos (y,x)
        print(channel_data2.shape)
        # transponieren [Nx, Ny]
        channel_data3 = np.transpose(channel_data2)
        #transpose dreht die Reihenfolge um
        print(channel_data3.shape)

        # anzeigen über matplotlib
        fig, ax = plt.subplots()
        im = ax.imshow(channel_data3, cmap='gray')
        plt.show()
        print('finished')
    else:
        continue

