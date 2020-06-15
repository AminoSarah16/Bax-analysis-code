"""
Finden der Segmentierungslinien im tif Bild (gelb: 255,255,0) und abspeichern als tiff file.

"""

import os
from PIL import Image  # https://pillow.readthedocs.io/en/stable/handbook/index.html
import matplotlib.pyplot as plt
from utils.utils import *

separation_color = np.array([255, 255, 0])  # gelb
file_suffix = 'ylwmask.jpg'

if __name__ == '__main__':

    # paths
    root_path = get_root_path()
    cell_separation_path = os.path.join(root_path, 'results', 'cell-separations')
    if not os.path.isdir(cell_separation_path):
        os.makedirs(cell_separation_path)

    # walk root path
    for (dirpath, dirnames, filenames) in os.walk(root_path):

        # all files ending on 'ylwmask.jpg'
        interesting_files = (file for file in filenames if file.endswith(file_suffix))

        # iterate over all interesting files
        for file in interesting_files:
            # load image and convert to numpy array
            file_path = os.path.join(dirpath, file)
            # print(os.path.isfile(file_path)) # only a check because one file was not readable
            if not os.path.isfile(file_path):
                print("file {} not found".format(file_path))
                continue
            img = np.array(Image.open(file_path))
            img = img.astype(np.float)

            # binarize according to separation color
            # maske = np.logical_not(np.logical_and.reduce((img[:, :, 0] == separation_color[0], img[:, :, 1] == separation_color[1], img[:, :, 2] == separation_color[2])))
            # fuzzy (because saved as jpg)
            distance = 0
            for i in range(3):
                distance += (img[:, :, i] - separation_color[i]) ** 2
            # display_image(distance > 4000, '')
            #plt.hist(distance.flatten(), bins=np.linspace(0, 200000, 200))
            #plt.show()
            maske = distance > 10000

            # label (https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html)
            labeled_mask, number_cells = ndimage.measurements.label(maske)

            # second round delete all the small ones
            objects = ndimage.measurements.find_objects(labeled_mask)
            # filter by cell area
            for i in range(number_cells):
                obj = objects[i]
                m = labeled_mask[obj[0], obj[1]]
                cell_pixelsum = np.sum(m == i + 1)
                if cell_pixelsum < 1000:
                    m[m == i + 1] = 0
                    labeled_mask[obj[0], obj[1]] = m
            labeled_mask, number_cells = ndimage.measurements.label(labeled_mask > 0)
            print('{} has {} cells'.format(file, number_cells))

            # check: display
            # display_image((img[:, :, 0], labeled_mask), ('', '{}'.format(number_cells)))

            # save as tif
            output_data = labeled_mask.astype(np.uint8)
            img = Image.fromarray(output_data)
            output_file = os.path.join(cell_separation_path, file[:-len(file_suffix)] + '.tif')
            img.save(output_file, format='tiff')  # schaut schwarz aus wegen LUT (in ImageJ laden und glasbey anwenden)