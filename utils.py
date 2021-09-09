import os
import os.path as op
from PIL import Image
import numpy as np
from sklearn.metrics import pairwise_distances
from nilearn import plotting
from neurosynth.base.dataset import download
from nimare.io import convert_neurosynth_to_dataset


def crop_image(image_fname):
    image = Image.open(image_fname)
    image.load()

    image_data = np.asarray(image)
    image_data_bw = image_data.mean(axis=2)
    non_empty_columns = np.where(image_data_bw.mean(axis=0) < 255)[0]
    non_empty_rows = np.where(image_data_bw.mean(axis=1) < 255)[0]
    cropBox = (
        min(non_empty_rows),
        max(non_empty_rows),
        min(non_empty_columns),
        max(non_empty_columns),
    )

    image_data_new = image_data[cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1, :]
    new_image = Image.fromarray(image_data_new)
    new_image.save(image_fname)


def insert(matrix, indices):
    matrix = np.insert(matrix, np.subtract(indices, np.arange(len(indices))), 0, axis=0)
    return matrix


def affinity(matrix, sparsity):
    # Generate percentile thresholds for 90th percentile
    perc = np.array([np.percentile(x, sparsity) for x in matrix])

    # Threshold each row of the matrix by setting values below 90th percentile to 0
    for i in range(matrix.shape[0]):
        matrix[i, matrix[i, :] < perc[i]] = 0
    matrix[matrix < 0] = 0

    # Now we are dealing with sparse vectors. Cosine similarity is used as affinity metric
    matrix = 1 - pairwise_distances(matrix, metric="cosine")

    return matrix


def neurosynth_download(ns_data_dir):

    dataset_file = op.join(ns_data_dir, "neurosynth_dataset.pkl.gz")

    os.makedirs(ns_data_dir, exist_ok=True)

    download(ns_data_dir, unpack=True)
    ###############################################################################
    # Convert Neurosynth database to NiMARE dataset file
    # --------------------------------------------------
    dset = convert_neurosynth_to_dataset(
        op.join(ns_data_dir, "database.txt"), op.join(ns_data_dir, "features.txt")
    )
    dset.save(dataset_file)


def plot_surfaces(grad_dict, index, outdir, prefix, normalize=False, cmap="jet"):
    grad_lh = grad_dict["grads_lh"][:, index]
    grad_rh = grad_dict["grads_rh"][:, index]
    if normalize:
        if np.max(grad_lh[grad_lh > 0]) > np.abs(np.min(grad_lh[grad_lh < 0])):
            grad_lh[grad_lh < 0] = grad_lh[grad_lh < 0] / np.abs(np.min(grad_lh[grad_lh < 0]))
            grad_lh[grad_lh < 0] = grad_lh[grad_lh < 0] * np.max(grad_lh[grad_lh > 0])
        else:
            grad_lh[grad_lh > 0] = grad_lh[grad_lh > 0] / np.max(grad_lh[grad_lh > 0])
            grad_lh[grad_lh > 0] = grad_lh[grad_lh > 0] * np.abs(np.min(grad_lh[grad_lh < 0]))
        if np.max(grad_rh[grad_rh > 0]) > np.abs(np.min(grad_rh[grad_rh < 0])):
            grad_rh[grad_rh < 0] = grad_rh[grad_rh < 0] / np.abs(np.min(grad_rh[grad_rh < 0]))
            grad_rh[grad_rh < 0] = grad_rh[grad_rh < 0] * np.max(grad_rh[grad_rh > 0])
        else:
            grad_rh[grad_rh > 0] = grad_rh[grad_rh > 0] / np.max(grad_rh[grad_rh > 0])
            grad_rh[grad_rh > 0] = grad_rh[grad_rh > 0] * np.abs(np.min(grad_rh[grad_rh < 0]))

    plotting.plot_surf_stat_map(
        grad_dict["pial_left"],
        grad_lh,
        hemi="left",
        bg_map=grad_dict["sulc_left"],
        bg_on_data=True,
        threshold=np.finfo(np.float32).eps,
        colorbar=False,
        view="medial",
        cmap=cmap,
        output_file=op.join(outdir, "{0}-{1}_left_medial.png".format(prefix, index)),
    )
    plotting.plot_surf_stat_map(
        grad_dict["pial_left"],
        grad_lh,
        hemi="left",
        bg_map=grad_dict["sulc_left"],
        bg_on_data=True,
        threshold=np.finfo(np.float32).eps,
        colorbar=False,
        view="lateral",
        cmap=cmap,
        output_file=op.join(outdir, "{0}-{1}_left_lateral.png".format(prefix, index)),
    )

    plotting.plot_surf_stat_map(
        grad_dict["pial_right"],
        grad_rh,
        hemi="right",
        bg_map=grad_dict["sulc_right"],
        bg_on_data=True,
        threshold=np.finfo(np.float32).eps,
        colorbar=False,
        view="medial",
        cmap=cmap,
        output_file=op.join(outdir, "{0}-{1}_right_medial.png".format(prefix, index)),
    )
    plotting.plot_surf_stat_map(
        grad_dict["pial_right"],
        grad_rh,
        hemi="right",
        bg_map=grad_dict["sulc_right"],
        bg_on_data=True,
        threshold=np.finfo(np.float32).eps,
        colorbar=True,
        view="lateral",
        cmap=cmap,
        output_file=op.join(outdir, "{0}-{1}_right_lateral.png".format(prefix, index)),
    )


def combine_plots(list_im, fname_out):

    for i in list_im:
        crop_image(i)

    imgs = [Image.open(i) for i in list_im]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack(
        (
            np.asarray(i.resize((int(i.size[0] * i.size[1] / imgs[3].size[1]), imgs[3].size[1])))
            for i in imgs
        )
    )
    # save that beautiful picture
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save(fname_out)
    for i in list_im:
        os.remove(i)
