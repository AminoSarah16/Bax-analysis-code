"""
Combines all the results from the cluster analysis.
"""

import os
import csv
from PIL import Image  # https://pillow.readthedocs.io/en/stable/handbook/index.html
from utils.utils import *

CLUSTER_EXTENSION = '.cluster.tiff'
MITO_EXTENSION = '.mito-mask.tiff'
CELL_NAMES = 'ABCDEFGH'
PIXEL_AREA = 0.015 * 0.015  # in µm
AREA_THRESHOLD = 0.2 * 0.2  # in µm


def get_condition(file):
    if 'Bax-wt' in file:
        return 'Bax_wt'
    if 'Bax-63-65A' in file:
        return 'Bax_63_65A'
    if 'Bax-H5i' in file:
        return 'Bax_H5i'
    if 'Bax-BH3i' in file:
        return 'Bax_Bh3i'
    raise RuntimeError('Unknown condition in {}'.format(file))


def get_replicate(file):
    if 'IF29_' in file or 'IF40_' in file:
        return 'replicate1'
    if 'IF36_' in file:
        return 'replicate2'
    if 'IF41.2_' in file or 'IF41_' in file:
        return 'replicate3'
    raise RuntimeError('Unknown replicate in {}'.format(file))


if __name__ == '__main__':

    # paths
    root_path = get_root_path()
    results_path = os.path.join(root_path, 'results')
    cell_separation_path = os.path.join(results_path, 'cell-separations')
    mito_path = os.path.join(results_path, 'mito-masks')
    bax_path = os.path.join(results_path, 'bax-structures')

    # get all clusters in the bax results path
    files = [file for file in os.listdir(bax_path) if file.endswith(CLUSTER_EXTENSION)]
    files.sort()
    print('analyse and combine {} measurements'.format(len(files)))

    # get all cell separation files
    cell_separation_files = os.listdir(cell_separation_path)

    # holding the bax total overlap results
    bax_total_overlap_results = []

    # holding the cluster area results
    bax_cluster_areas = {}

    # holding the cluster overlap results
    bax_cluster_overlaps = {}

    # loop over files
    # files = files[:2] # just for debugging
    for file in files:
        condition = get_condition(file)
        replicate = get_replicate(file)
        print('\nwork on {} {} {}'.format(condition, replicate, file))
        file_name = file[:-len(CLUSTER_EXTENSION)]
        # file body is everything until after _cl (used to associate cell masks)
        idx = file_name.find('_cl')
        idx2 = file_name.find('_', idx+1)
        if idx2 == -1:
            idx2 = file_name.find('-', idx+1)
        file_body = file_name[:idx2]
        print(' file body {}'.format(file_body))

        # load cluster image
        all_clusters = np.array(Image.open(os.path.join(bax_path, file)))

        # search for corresponding mito mask
        mito_file = os.path.join(mito_path, file_name + MITO_EXTENSION)
        if not os.path.isfile(mito_file):
            print(' mito mask not existing for {}'.format(file))
            RuntimeError('Please create mito mask')
        all_mitos = np.array(Image.open(mito_file))

        # search for corresponding cell separation
        cell_separation_file = None
        for name in cell_separation_files:
            if name.startswith(file_body + '_') or name.startswith(file_body + '-'):
                cell_separation_file = name
                break
        if not cell_separation_file:
            print(' no cell separation file, assume single cell')
            cells = np.ones(all_clusters.shape) # 1 everywhere
        else:
            cell_separation_files.remove(cell_separation_file)
            print(' use cell separation {}'.format(cell_separation_file))
            cells = np.array(Image.open(os.path.join(cell_separation_path, cell_separation_file)))
        number_cells = int(np.max(cells))

        # everything is loaded, we can start now

        # iterate over cells
        for c in range(1, number_cells + 1):
            cell_name = CELL_NAMES[c - 1]
            print(' cell {}'.format(cell_name))

            # cell mask
            cell_mask = cells == c

            # first overlap bax clusters total and mito per cell
            cell_clusters = all_clusters & cell_mask
            cell_mitos = all_mitos & cell_mask

            cell_cluster_area = np.sum(cell_clusters)
            cell_mito_area = np.sum(cell_mitos)
            cell_cluster_and_mito_area = np.sum(cell_clusters & cell_mitos)
            bax_mito_overlap = cell_cluster_and_mito_area / cell_cluster_area
            mito_bax_overlap = cell_cluster_and_mito_area / cell_mito_area

            # check: display
            # display_image((cell, cl, mt), ('cell', 'cluster in cell', 'mito in cell'))

            print('  bax mito overlap {:.3f}, mito bax overlap {:.3f}'.format(bax_mito_overlap, mito_bax_overlap))

            # store total overlap results
            bax_total_overlap_results.append((condition, replicate, file_body, cell_name, bax_mito_overlap, mito_bax_overlap))

            # get single bax cluster sizes
            clusters, number_clusters = ndimage.measurements.label(cell_clusters)
            objects = ndimage.measurements.find_objects(clusters)
            cluster_areas = np.zeros((number_clusters, 1))
            cluster_overlaps = []
            for i in range(number_clusters):
                obj = objects[i]
                m = clusters[obj[0], obj[1]] == i + 1
                cluster_area = np.sum(m) * PIXEL_AREA
                cluster_areas[i] = cluster_area
                if cluster_area > AREA_THRESHOLD:
                    # compute mito overlap
                    mti = cell_mitos[obj[0], obj[1]]
                    overlap = np.sum(m & mti) / np.sum(m)
                    cluster_overlaps.append(overlap)
            cluster_overlaps = np.vstack(cluster_overlaps)
            print('  {} clusters in cell, {} clusters above treshold'.format(number_clusters, np.sum(cluster_areas > AREA_THRESHOLD)))

            # write areas to csv file
            results_file = os.path.join(bax_path, '{}-{}-cluster-areas.csv'.format(file_body, cell_name))
            with open(results_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(cluster_areas)

            # write overlaps to csv
            results_file = os.path.join(bax_path, '{}-{}-large-cluster-overlaps.csv'.format(file_body, cell_name))
            with open(results_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(cluster_overlaps)

            # add areas to dictionary
            key = '{}-{}'.format(condition, replicate)
            if key in bax_cluster_areas:
                bax_cluster_areas[key].append(cluster_areas)
            else:
                bax_cluster_areas[key] = [cluster_areas]

            # add overlaps to dictionary
            if key in bax_cluster_overlaps:
                bax_cluster_overlaps[key].append(cluster_overlaps)
            else:
                bax_cluster_overlaps[key] = [cluster_overlaps]

    if cell_separation_files:
        print('unused cell separation files {}'.format(cell_separation_files))

    # sort results by condition, replicate, file name, cell
    bax_total_overlap_results.sort(key=lambda x: ''.join(x[:4]))

    # write bax total overlap to csv file
    results_file = os.path.join(results_path, 'bax-cluster-total-mito-overlap.csv')
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('Condition', 'Replicate', 'File', 'Cell', 'Fraction of Bax on Mito', 'Fraction of Mito on Bax'))
        writer.writerows(bax_total_overlap_results)

    # write out bax cluster areas
    for k, v in sorted(bax_cluster_areas.items(), key=lambda x: x[0]):
        results_file = os.path.join(results_path, 'bax-cluster-areas-{}.csv'.format(k))
        # v = np.stack(v, axis=0)
        v = np.vstack(v)
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(v)
        # print how many are above/below the threshold
        print('cluster areas {} - {:.2f}% above threshold'.format(k, np.mean(v > AREA_THRESHOLD) * 100))

    # write out bax cluster overlaps
    for k, v in sorted(bax_cluster_areas.items(), key=lambda x: x[0]):
        results_file = os.path.join(results_path, 'bax-cluster-overlaps-{}.csv'.format(k))
        # v = np.stack(v, axis=0)
        v = np.vstack(v)
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(v)
        # print average overlap and how many have overlap
        print('large cluster overlaps {} - avg. overlap {:.2f}%, {:.2f}% with some kind of overlap'.format(k, np.mean(v)*100, np.mean(v>0)*100))
