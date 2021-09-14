"""
Based on
https://github.com/BIDS-Apps/example/blob/aa0d4808974d79c9fbe54d56d3b47bb2cf4e0a0d/run.py
"""
import argparse
import os
import os.path as op
import pickle
import shutil

import nibabel as nib
import numpy as np
from mapalign import embed
from neurosynth.base.dataset import download
from nilearn import datasets, masking, plotting, surface
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.input_data import NiftiLabelsMasker
from nimare.dataset import Dataset
from nimare.io import convert_neurosynth_to_dataset, convert_sleuth_to_dataset
from nimare.meta.kernel import ALEKernel, Peaks2MapsKernel
from nimare.transforms import vox2mm
from sklearn.metrics import pairwise_distances


def plot_surfaces(grad_dict, index, outdir):
    grad_lh = grad_dict["grads_lh"][:, index]
    if np.max(grad_lh[grad_lh > 0]) > np.abs(np.min(grad_lh[grad_lh < 0])):
        grad_lh[grad_lh < 0] = grad_lh[grad_lh < 0] / np.abs(np.min(grad_lh[grad_lh < 0]))
        grad_lh[grad_lh < 0] = grad_lh[grad_lh < 0] * np.max(grad_lh[grad_lh > 0])
    else:
        grad_lh[grad_lh > 0] = grad_lh[grad_lh > 0] / np.max(grad_lh[grad_lh > 0])
        grad_lh[grad_lh > 0] = grad_lh[grad_lh > 0] * np.abs(np.min(grad_lh[grad_lh < 0]))
    plotting.plot_surf_stat_map(
        grad_dict["pial_left"],
        grad_lh,
        hemi="left",
        bg_map=grad_dict["sulc_left"],
        bg_on_data=True,
        colorbar=False,
        view="medial",
        cmap="jet",
        output_file=op.join(outdir, "gradient-{0}_left_medial.png".format(index)),
    )
    plotting.plot_surf_stat_map(
        grad_dict["pial_left"],
        grad_lh,
        hemi="left",
        bg_map=grad_dict["sulc_left"],
        bg_on_data=True,
        colorbar=False,
        view="lateral",
        cmap="jet",
        output_file=op.join(outdir, "gradient-{0}_left_lateral.png".format(index)),
    )
    grad_rh = grad_dict["grads_rh"][:, index]
    if np.max(grad_rh[grad_rh > 0]) > np.abs(np.min(grad_rh[grad_rh < 0])):
        grad_rh[grad_rh < 0] = grad_rh[grad_rh < 0] / np.abs(np.min(grad_rh[grad_rh < 0]))
        grad_rh[grad_rh < 0] = grad_rh[grad_rh < 0] * np.max(grad_rh[grad_rh > 0])
    else:
        grad_rh[grad_rh > 0] = grad_rh[grad_rh > 0] / np.max(grad_rh[grad_rh > 0])
        grad_rh[grad_rh > 0] = grad_rh[grad_rh > 0] * np.abs(np.min(grad_rh[grad_rh < 0]))
    plotting.plot_surf_stat_map(
        grad_dict["pial_right"],
        grad_rh,
        hemi="right",
        bg_map=grad_dict["sulc_right"],
        bg_on_data=True,
        colorbar=False,
        view="medial",
        cmap="jet",
        output_file=op.join(outdir, "gradient-{0}_right_medial.png".format(index)),
    )
    plotting.plot_surf_stat_map(
        grad_dict["pial_right"],
        grad_rh,
        hemi="right",
        bg_map=grad_dict["sulc_right"],
        bg_on_data=True,
        colorbar=True,
        view="lateral",
        cmap="jet",
        output_file=op.join(outdir, "gradient-{0}_right_lateral.png".format(index)),
    )


def calculate_affinity(matrix, sparsity):
    # Generate percentile thresholds for 90th percentile
    perc = np.array([np.percentile(x, sparsity) for x in matrix])
    # Threshold each row of the matrix by setting values below 90th percentile to 0
    for i in range(matrix.shape[0]):
        matrix[i, matrix[i, :] < perc[i]] = 0
    matrix[matrix < 0] = 0

    # Now we are dealing with sparse vectors. Cosine similarity is used as affinity metric

    matrix = 1 - pairwise_distances(matrix, metric="cosine")
    print("Done calculating affinity matrix")

    return matrix


def build_macms(dset, imgs, coords):
    macms = []
    inds_discard = []
    macms = np.zeros((np.shape(coords)[0], np.shape(imgs)[1]))
    for i in range(coords.shape[0]):
        print(i)
        tmpids = dset.get_studies_by_coordinate(np.expand_dims(coords[i], axis=1).T, r=12)
        if len(tmpids) > 0:
            xy, x_ind, y_ind = np.intersect1d(tmpids, dset.ids, return_indices=True)
            macms[i, :] = 1.0 - np.prod(1.0 - imgs[y_ind, :], axis=0)
            del xy, x_ind, y_ind
        else:
            inds_discard.append(i)
        del tmpids

    if len(inds_discard) > 0:
        macms = np.delete(macms, inds_discard, axis=0)
    return macms, inds_discard


def insert(matrix, indices):
    matrix = np.insert(matrix, np.subtract(indices, np.arange(len(indices))), 0, axis=0)
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


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "This script will generate axials, surface medial and surface lateral view images "
            "with the specified overlay."
        )
    )
    parser.add_argument(
        "--neurosynth",
        required=False,
        dest="neurosynth",
        action="store_true",
        help=("Query the Neurosynth database."),
    )
    parser.add_argument(
        "--subcortical",
        required=False,
        dest="subcort",
        action="store_true",
        default=False,
        help=("Whether to include the subcortical voxels."),
    )
    parser.add_argument(
        "--nimare-dataset",
        required=False,
        dest="nimare_dataset",
        default=None,
        help=("Import a NiMARE dataset."),
    )
    parser.add_argument(
        "--neurosynth-file",
        required=False,
        dest="neurosynth_file",
        help="Full path to neurosynth file to use as database.",
    )
    parser.add_argument(
        "--sleuth-file",
        required=False,
        dest="sleuth_file",
        help="Full path to sleuth file to use as database.",
    )
    parser.add_argument(
        "--roi-mask",
        required=False,
        dest="roi_mask",
        help="Full path to roi mask for selecting studies.",
    )
    parser.add_argument(
        "--approach",
        required=False,
        dest="approach",
        default="dm",
        help="Embedding approach for gradients.",
    )
    parser.add_argument(
        "--affinity",
        required=False,
        dest="affinity",
        default="cosine",
        help="Kernel function to build the affinity matrix.",
    )
    parser.add_argument(
        "--term",
        required=False,
        dest="term",
        help="Term or list of terms (e.g. ['load', 'rest'] for selecting studies.",
    )
    parser.add_argument(
        "--topic",
        required=False,
        dest="topic",
        nargs="*",
        help="Topic or list of topics (e.g. ['topic002', 'topic023'] for selecting studies.",
    )
    parser.add_argument(
        "--kernel",
        required=False,
        dest="kernel",
        default="alekernel",
        help="Kernel for converting peaks.",
    )
    parser.add_argument(
        "--atlas",
        required=False,
        dest="atlas",
        default="fsaverage5",
        help=(
            "Atlas name for parcellating data: harvard-oxford, aal, craddock-2012, "
            "destrieux-2009, msdl, fsaverage5 (surface), hcp (surface)"
        ),
    )
    parser.add_argument(
        "--gradients",
        required=False,
        dest="gradients",
        default=None,
        help="Number of gradients to produce.",
    )
    parser.add_argument(
        "--sparsity",
        required=False,
        dest="sparsity",
        default=0.9,
        help="Sparsity for thresholding connectivity matrix.",
    )
    parser.add_argument(
        "-w", "--workdir", required=False, dest="workdir", help="Path to working directory."
    )
    parser.add_argument(
        "-o", "--outdir", required=False, dest="outdir", help="Path to output directory."
    )
    return parser


def _main(argv=None):
    args = get_parser().parse_args(argv)
    args = vars(args)
    main(**args)


def main(
    workdir,
    outdir,
    atlas,
    kernel,
    sparsity,
    affinity,
    approach,
    gradients,
    subcort,
    neurosynth,
    neurosynth_file,
    sleuth_file,
    nimare_dataset,
    roi_mask,
    term,
    topic,
):
    workdir = op.join(workdir, "tmp")
    if op.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)

    atlas_name = "atlas-{0}".format(atlas)
    kernel_name = "kernel-{0}".format(kernel)
    sparsity_name = "sparsity-{0}".format(sparsity)
    affinity_name = "affinity-{0}".format(affinity)
    approach_name = "approach-{0}".format(approach)
    gradients_name = "gradients-{0}".format(gradients)
    dset = None

    # handle neurosynth dataset, if called
    if neurosynth:
        if neurosynth_file is None:

            ns_data_dir = op.join(workdir, "neurosynth")
            dataset_file = op.join(ns_data_dir, "neurosynth_dataset.pkl.gz")
            # download neurosynth dataset if necessary
            if not op.isfile(dataset_file):
                neurosynth_download(ns_data_dir)

        else:
            dataset_file = neurosynth_file

        dset = Dataset.load(dataset_file)
        dataset_name = "dataset-neurosynth"

    # handle sleuth text file, if called
    if sleuth_file is not None:
        dset = convert_sleuth_to_dataset(sleuth_file, target="mni152_2mm")
        dataset_name = "dataset-{0}".format(op.basename(sleuth_file).split(".")[0])

    if nimare_dataset is not None:
        dset = Dataset.load(nimare_dataset)
        dataset_name = "dataset-{0}".format(op.basename(nimare_dataset).split(".")[0])

    if dset:
        # slice studies, if needed
        if roi_mask is not None:
            roi_ids = dset.get_studies_by_mask(roi_mask)
            print(
                "{}/{} studies report at least one coordinate in the "
                "ROI".format(len(roi_ids), len(dset.ids))
            )
            dset_sel = dset.slice(roi_ids)
            dset = dset_sel
            dataset_name = "dataset-neurosynth_mask-{0}".format(
                op.basename(roi_mask).split(".")[0]
            )

        if term is not None:
            labels = ["Neurosynth_TFIDF__{label}".format(label=label) for label in [term]]
            term_ids = dset.get_studies_by_label(labels=labels, label_threshold=0.1)
            print(
                "{}/{} studies report association "
                "with the term {}".format(len(term_ids), len(dset.ids), term)
            )
            dset_sel = dset.slice(term_ids)
            dset = dset_sel
            # img_inds = np.nonzero(dset.masker.mask_img.get_fdata())  # unused
            # vox_locs = np.unravel_index(img_inds, dset.masker.mask_img.shape)  # unused
            dataset_name = "dataset-neurosynth_term-{0}".format(term)

        if topic is not None:
            topics = [
                "Neurosynth_{version}__{topic}".format(version=topic[0], topic=topic)
                for topic in topic[1:]
            ]
            topics_ids = []
            for topic in topics:
                topic_ids = dset.annotations.id[np.where(dset.annotations[topic])[0]].tolist()
                topics_ids.extend(topic_ids)
                print(
                    "{}/{} studies report association "
                    "with the term {}".format(len(topic_ids), len(dset.ids), topic)
                )
            topics_ids_unique = np.unique(topics_ids)
            print("{} unique ids".format(len(topics_ids_unique)))
            dset_sel = dset.slice(topics_ids_unique)
            dset = dset_sel
            # img_inds = np.nonzero(dset.masker.mask_img.get_fdata())  # unused
            # vox_locs = np.unravel_index(img_inds, dset.masker.mask_img.shape)  # unused
            dataset_name = "dataset-neurosynth_topic-{0}".format("_".join(topic[1:]))

        if (
            neurosynth
            or (sleuth_file is not None)
            or (nimare_dataset is not None)
        ):
            # set kernel for MA smoothing
            if kernel == "peaks2maps":
                print("Running peak2maps")
                k = Peaks2MapsKernel(resample_to_mask=True)
            elif kernel == "alekernel":
                print("Running alekernel")
                k = ALEKernel(fwhm=15)

            if atlas is not None:
                if atlas == "harvard-oxford":
                    print("Parcellating using the Harvard Oxford Atlas")
                    # atlas_labels = atlas.labels[1:]  # unused
                    atlas_shape = atlas.maps.shape
                    atlas_affine = atlas.maps.affine
                    atlas_data = atlas.maps.get_fdata()
                elif atlas == "aal":
                    print("Parcellating using the AAL Atlas")
                    atlas = datasets.fetch_atlas_aal()
                    # atlas_labels = atlas.labels  # unused
                    atlas_shape = nib.load(atlas.maps).shape
                    atlas_affine = nib.load(atlas.maps).affine
                    atlas_data = nib.load(atlas.maps).get_fdata()
                elif atlas == "craddock-2012":
                    print("Parcellating using the Craddock-2012 Atlas")
                    atlas = datasets.fetch_atlas_craddock_2012()
                elif atlas == "destrieux-2009":
                    print("Parcellating using the Destrieux-2009 Atlas")
                    atlas = datasets.fetch_atlas_destrieux_2009(lateralized=True)
                    # atlas_labels = atlas.labels[3:]  # unused
                    atlas_shape = nib.load(atlas.maps).shape
                    atlas_affine = nib.load(atlas.maps).affine
                    atlas_data = nib.load(atlas.maps).get_fdata()
                elif atlas == "msdl":
                    print("Parcellating using the MSDL Atlas")
                    atlas = datasets.fetch_atlas_msdl()
                elif atlas == "surface":
                    print("Generating surface vertices")

                if atlas != "fsaverage5" and atlas != "hcp":
                    imgs = k.transform(dset, return_type="image")

                    masker = NiftiLabelsMasker(
                        labels_img=atlas.maps, standardize=True, memory="nilearn_cache"
                    )
                    time_series = masker.fit_transform(imgs)

                else:
                    # change to array for other approach
                    imgs = k.transform(dset, return_type="image")
                    print(np.shape(imgs))

                    if atlas == "fsaverage5":
                        fsaverage = fetch_surf_fsaverage(mesh="fsaverage5")
                        pial_left = fsaverage.pial_left
                        pial_right = fsaverage.pial_right
                        medial_wall_inds_left = surface.load_surf_data(
                            "./templates/lh.Medial_wall.label"
                        )
                        print(np.shape(medial_wall_inds_left))
                        medial_wall_inds_right = surface.load_surf_data(
                            "./templates/rh.Medial_wall.label"
                        )
                        print(np.shape(medial_wall_inds_right))
                        sulc_left = fsaverage.sulc_left
                        sulc_right = fsaverage.sulc_right

                    elif atlas == "hcp":
                        pial_left = "./templates/S1200.L.pial_MSMAll.32k_fs_LR.surf.gii"
                        pial_right = "./templates/S1200.R.pial_MSMAll.32k_fs_LR.surf.gii"
                        medial_wall_inds_left = np.where(
                            nib.load("./templates/hcp.tmp.lh.dscalar.nii").get_fdata()[0] == 0
                        )[0]
                        medial_wall_inds_right = np.where(
                            nib.load("./templates/hcp.tmp.rh.dscalar.nii").get_fdata()[0] == 0
                        )[0]
                        left_verts = 32492 - len(medial_wall_inds_left)
                        sulc_left = nib.load(
                            "./templates/S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii"
                        ).get_fdata()[0][0:left_verts]
                        sulc_left = np.insert(
                            sulc_left,
                            np.subtract(
                                medial_wall_inds_left, np.arange(len(medial_wall_inds_left))
                            ),
                            0,
                        )
                        sulc_right = nib.load(
                            "./templates/S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii"
                        ).get_fdata()[0][left_verts:]
                        sulc_right = np.insert(
                            sulc_right,
                            np.subtract(
                                medial_wall_inds_right, np.arange(len(medial_wall_inds_right))
                            ),
                            0,
                        )

                    surf_lh = surface.vol_to_surf(
                        imgs,
                        pial_left,
                        radius=6.0,
                        interpolation="nearest",
                        kind="ball",
                        n_samples=None,
                        mask_img=dset.masker.mask_img,
                    )
                    surf_rh = surface.vol_to_surf(
                        imgs,
                        pial_right,
                        radius=6.0,
                        interpolation="nearest",
                        kind="ball",
                        n_samples=None,
                        mask_img=dset.masker.mask_img,
                    )
                    surfs = np.transpose(np.vstack((surf_lh, surf_rh)))
                    del surf_lh, surf_rh

                    # handle cortex first
                    coords_left = surface.load_surf_data(pial_left)[0]
                    coords_left = np.delete(coords_left, medial_wall_inds_left, axis=0)
                    coords_right = surface.load_surf_data(pial_right)[0]
                    coords_right = np.delete(coords_right, medial_wall_inds_right, axis=0)

                    print("Left Hemipshere Vertices")
                    surface_macms_lh, inds_discard_lh = build_macms(dset, surfs, coords_left)
                    print(np.shape(surface_macms_lh))
                    print(inds_discard_lh)

                    print("Right Hemipshere Vertices")
                    surface_macms_rh, inds_discard_rh = build_macms(dset, surfs, coords_right)
                    print(np.shape(surface_macms_rh))
                    print(len(inds_discard_rh))

                    lh_vertices_total = np.shape(surface_macms_lh)[0]
                    rh_vertices_total = np.shape(surface_macms_rh)[0]
                    time_series = np.transpose(np.vstack((surface_macms_lh, surface_macms_rh)))
                    print(np.shape(time_series))
                    del surface_macms_lh, surface_macms_rh

                    if subcort:
                        subcort_img = nib.load("templates/rois-subcortical_mni152_mask.nii.gz")
                        subcort_vox = np.asarray(np.where(subcort_img.get_fdata()))
                        subcort_mm = vox2mm(subcort_vox.T, subcort_img.affine)

                        print("Subcortical Voxels")
                        subcort_macm, inds_discard_subcort = build_macms(dset, surfs, subcort_mm)

                        num_subcort_vox = np.shape(subcort_macm)[0]
                        print(inds_discard_subcort)

                        time_series = np.hstack((time_series, np.asarray(subcort_macm).T))
                        print(np.shape(time_series))

                time_series = time_series.astype("float32")

                print("calculating correlation matrix")
                correlation = ConnectivityMeasure(kind="correlation")
                time_series = correlation.fit_transform([time_series])[0]
                print(np.shape(time_series))

                if affinity == "cosine":
                    time_series = calculate_affinity(time_series, 10 * sparsity)

            else:
                time_series = np.transpose(k.transform(dset, return_type="array"))

    print("Performing gradient analysis")

    gradients, statistics = embed.compute_diffusion_map(
        time_series, alpha=0.5, return_result=True, overwrite=True
    )
    pickle.dump(statistics, open(op.join(workdir, "statistics.p"), "wb"))

    # if subcortical included in gradient decomposition, remove gradient scores
    if subcort:
        subcort_grads = gradients[np.shape(gradients)[0] - num_subcort_vox :, :]
        subcort_grads = insert(subcort_grads, inds_discard_subcort)
        gradients = gradients[0 : np.shape(gradients)[0] - num_subcort_vox, :]

    # get left hemisphere gradient scores, and insert 0's where medial wall is
    gradients_lh = gradients[0:lh_vertices_total, :]
    if len(inds_discard_lh) > 0:
        gradients_lh = insert(gradients_lh, inds_discard_lh)
    gradients_lh = insert(gradients_lh, medial_wall_inds_left)

    # get right hemisphere gradient scores and insert 0's where medial wall is
    gradients_rh = gradients[-rh_vertices_total:, :]
    if len(inds_discard_rh) > 0:
        gradients_rh = insert(gradients_rh, inds_discard_rh)
    gradients_rh = insert(gradients_rh, medial_wall_inds_right)

    grad_dict = {
        "grads_lh": gradients_lh,
        "grads_rh": gradients_rh,
        "pial_left": pial_left,
        "sulc_left": sulc_left,
        "pial_right": pial_right,
        "sulc_right": sulc_right,
    }
    if subcort:
        grad_dict["subcort_grads"] = subcort_grads
    pickle.dump(grad_dict, open(op.join(workdir, "gradients.p"), "wb"))

    # map the gradient to the parcels
    for i in range(np.shape(gradients)[1]):
        if atlas is not None:
            if atlas == "fsaverage5" or atlas == "hcp":

                plot_surfaces(grad_dict, i, workdir)

                if subcort:
                    tmpimg = masking.unmask(subcort_grads[:, i], subcort_img)
                    nib.save(tmpimg, op.join(workdir, "gradient-{0}.nii.gz".format(i)))
            else:
                tmpimg = np.zeros(atlas_shape)
                for j, n in enumerate(np.unique(atlas_data)[1:]):
                    inds = atlas_data == n
                    tmpimg[inds] = gradients[j, i]
                    nib.save(
                        nib.Nifti1Image(tmpimg, atlas_affine),
                        op.join(workdir, "gradient-{0}.nii.gz".format(i)),
                    )
        else:
            tmpimg = np.zeros(np.prod(dset.masker.mask_img.shape))
            inds = np.ravel_multi_index(
                np.nonzero(dset.masker.mask_img.get_fdata()), dset.masker.mask_img.shape
            )
            tmpimg[inds] = gradients[:, i]
            nib.save(
                nib.Nifti1Image(
                    np.reshape(tmpimg, dset.masker.mask_img.shape), dset.masker.mask_img.affine
                ),
                op.join(workdir, "gradient-{0}.nii.gz".format(i)),
            )

            os.system(
                "python3 /Users/miriedel/Desktop/GitHub/surflay/make_figures.py "
                "-f {grad_image} --colormap jet".format(
                    grad_image=op.join(workdir, "gradient-{0}.nii.gz".format(i))
                )
            )

    output_dir = op.join(
        outdir,
        (
            f"{dataset_name}_{atlas_name}_{kernel_name}_{sparsity_name}_{gradients_name}_"
            f"{affinity_name}_{approach_name}"
        )
    )

    shutil.copytree(workdir, output_dir)

    shutil.rmtree(workdir)


if __name__ == "__main__":
    _main()
