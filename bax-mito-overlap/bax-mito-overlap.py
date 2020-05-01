"""
Berechne Überlapp zwischen Mito-Kanal und Bax Strukturen
"""

from PIL import Image  # https://pillow.readthedocs.io/en/stable/handbook/index.html
import os
import numpy as np
from utils.utils import *

def compute_smallest_distance(x, y, mito_mask):
    """
    Berechnet des kleinsten Abstand von der Position (x,y) zu irgendereiner Position in mito_mask, bei der mito_mask gesetzt ist
    """

    Dmax = 100  # 100 pixel is maximaler abstand

    x1 = int(max(0, np.floor(x)-Dmax))
    x2 = int(min(mito_mask.shape[0], np.ceil(x)+Dmax))
    y1 = int(max(0, np.floor(y)-Dmax))
    y2 = int(min(mito_mask.shape[1], np.ceil(y)+Dmax))

    # iteriere über alle Positionen in der Nähe von (x,y) und berechne Abstände, merke den kleinsten
    Dmin = Dmax
    for xi in range(x1, x2):
        for yi in range(y1, y2):
            if mito_mask[xi, yi]:
                D = np.sqrt((x-xi)*(x-xi)+(y-yi)*(y-yi))
                if D < Dmin:
                    Dmin = D

    # set all with distance < 1 auf 0
    if Dmin < 1:
        Dmin = 0

    return Dmin




def compute_things(mito_mask, bax_clusters, bax_structures, bax_statistics):

    # anzeige
    # display_image((mito_mask, bax_clusters, bax_structures), ('Mito Maske', 'Bax Cluster', 'Bax Structures'))

    # x, y positions
    size = mito_mask.shape
    xi, yi = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')

    # iteriere über cluster und berechne überlapp
    Nc = np.amax(bax_clusters)
    for i in range(Nc):
        # get x and y positions of a cluster
        m = bax_clusters == i + 1 # weil im Bild der erste Cluster den Wert 1 hat
        xj = xi[m]
        yj = yi[m]

        # überlapp berechnen
        N1 = xj.size
        N2 = 0
        for j in range(N1):
            if mito_mask[xj[j], yj[j]] > 0:
                N2 += 1
        mito_overlap = N2 / N1

        # schwerpunkt berechnen
        xm = np.mean(xj)
        ym = np.mean(yj)

        # distanz schwerpunkt - mito berechnen
        mito_distance = compute_smallest_distance(xm, ym, mito_mask)

        print((i, mito_overlap, mito_distance))


if __name__ == '__main__':

    root_path = r'C:\Users\Sarah\Documents\Python\Bax-analysis\IF36_selected-for-analysis-with-Jan'
    mask_path = os.path.join(root_path, 'results', 'mito-masks')
    bax_path = os.path.join(root_path, 'results', 'bax-structures')

    for filename in os.listdir(root_path):  # ich erstelle eine Liste mit den Filenames in dem Ordner
        if filename.endswith(".msr"):  # wenn die Endung .msr ist, dann mach was damit, nämlich:
            print(filename)

            # gibt es eine Mito-Maske für diese Messung?
            path = os.path.join(mask_path, filename[:-4] + '.tiff')
            if not os.path.isfile(path):
                # keine mito maske, continue
                continue

            # einlesen der mitomaske
            im = Image.open(path)
            mito_mask = np.array(im)

            # gibt es eine Bax Struktur für diese Messung?
            path = os.path.join(bax_path, filename[:-4] + '.structures.csv')
            if not os.path.isfile(path):
                # keine bax structure, continue
                continue

            # einlesen der bax struktur
            path = os.path.join(bax_path, filename[:-4] + '.cluster.tiff')
            im = Image.open(path)
            bax_clusters = np.array(im)

            path = os.path.join(bax_path, filename[:-4] + '.structures-id.tiff')
            im = Image.open(path)
            bax_structures = np.array(im)

            path = os.path.join(bax_path, filename[:-4] + '.structures.csv')
            bax_statistics = np.loadtxt(path, delimiter=',')

            # compute overlap
            compute_things(mito_mask, bax_clusters, bax_structures, bax_statistics)
