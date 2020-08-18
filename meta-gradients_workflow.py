"""
Based on
https://github.com/BIDS-Apps/example/blob/aa0d4808974d79c9fbe54d56d3b47bb2cf4e0a0d/run.py
"""
import os
import os.path as op
import argparse
import nibabel as nib
import numpy as np
import nimare
import shutil
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn import datasets
from nimare.io import convert_sleuth_to_dataset
from neurosynth.base.dataset import download
from nimare.io import convert_neurosynth_to_dataset
from brainspace.gradient import GradientMaps


def neurosynth_download(ns_data_dir):

    dataset_file = op.join(ns_data_dir, 'neurosynth_dataset.pkl.gz')

    os.makedirs(ns_data_dir, exist_ok=True)

    download(ns_data_dir, unpack=True)
    ###############################################################################
    # Convert Neurosynth database to NiMARE dataset file
    # --------------------------------------------------
    dset = convert_neurosynth_to_dataset(
        op.join(ns_data_dir, 'database.txt'),
        op.join(ns_data_dir, 'features.txt'))
    dset.save(dataset_file)


def get_parser():
    parser = argparse.ArgumentParser(description='This script will generate axials, surface medial and surface lateral view images with the specified overlay.')
    parser.add_argument('--neurosynth', required=False, dest='neurosynth', action='store_true',
                        help=('Query the Neurosynth database.'))
    parser.add_argument('--neurosynth-file', required=False, dest='neurosynth_file',
                        help='Full path to neurosynth file to use as database.')
    parser.add_argument('--sleuth-file', required=False, dest='sleuth_file',
                        help='Full path to sleuth file to use as database.')
    parser.add_argument('--roi-mask', required=False, dest='roi_mask',
                        help='Full path to roi mask for selecting studies.')
    parser.add_argument('--approach', required=False, dest='approach', default='dm',
                        help='Embedding approach for gradients.')
    parser.add_argument('--affinity-kernel', required=False, dest='affinity_kernel', default=None,
                        help='Kernel function to build the affinity matrix.')
    parser.add_argument('--term', required=False, dest='term',
                        help='Term or list of terms (e.g. [\'load\', \'rest\'] for selecting studies.')
    parser.add_argument('--kernel', required=False, dest='kernel', default=None,
                        help='Kernel for converting peaks.')
    parser.add_argument('--fmri-data', required=False, dest='fmri_data',
                        help='Full path to fMRI data.')
    parser.add_argument('--atlas', required=False, dest='atlas', default=None,
                        help='Atlas name for parcellating data: harvard-oxford, aal, craddock-2012, destrieux-2009, msdl.')
    parser.add_argument('--prefix', required=False, dest='prefix',
                        help='prefix name.')
    parser.add_argument('-w', '--workdir', required=False, dest='workdir',
                        help='Path to working directory.')
    parser.add_argument('-o', '--outdir', required=False, dest='outdir',
                        help='Path to output directory.')
    return parser


def main(argv=None):

    args = get_parser().parse_args(argv)

    workdir = op.join(args.workdir, 'tmp')
    if op.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)

    atlas_name = 'atlas-{0}'.format(args.atlas)
    kernel_name = 'kernel-{0}'.format(args.kernel)

    #handle neurosynth dataset, if called
    if args.neurosynth == True:
        from nimare.dataset import Dataset
        if args.neurosynth_file is None:

            ns_data_dir = op.join(workdir, 'neurosynth')
            dataset_file = op.join(ns_data_dir, 'neurosynth_dataset.pkl.gz')
            # download neurosynth dataset if necessary
            if not op.isfile(dataset_file):
                neurosynth_download(ns_data_dir)

        else:
            dataset_file = args.neurosynth_file

        dset = Dataset.load(dataset_file)

    #handle sleuth text file, if called
    if args.sleuth_file is not None:
        dset = convert_sleuth_to_dataset(args.sleuth_file, target="mni152_2mm")
        dataset_name = 'dataset-{0}'.format(op.basename(args.sleuth_file).split('.')[0])

    #slice studies, if needed
    if args.roi_mask is not None:
        roi_ids = dset.get_studies_by_mask(args.roi_mask)
        print('{}/{} studies report at least one coordinate in the '
            'ROI'.format(len(roi_ids), len(dset.ids)))
        dset_sel = dset.slice(roi_ids)
        dset = dset_sel
        dataset_name = 'dataset-neurosynth_mask-{0}'.format(op.basename(args.roi_mask).split('.')[0])

    if args.term is not None:
        labels = ['Neurosynth_TFIDF__{label}'.format(label=label) for label in [args.term]]
        term_ids = dset.get_studies_by_label(labels=labels, label_threshold=0.1)
        print('{}/{} studies report association '
            'with the term {}'.format(len(term_ids), len(dset.ids), args.term))
        dset_sel = dset.slice(term_ids)
        dset = dset_sel
        img_inds = np.nonzero(dset.masker.mask_img.get_fdata())
        vox_locs = np.unravel_index(img_inds, dset.masker.mask_img.shape)
        dataset_name = 'dataset-neurosynth_term-{0}'.format(args.term)

    if (args.neurosynth == True) or (args.sleuth_file is not None):
        if args.kernel == 'peaks2maps':
            from nimare.meta.kernel import Peaks2MapsKernel
            print("Running peak2maps")
            k = Peaks2MapsKernel(resample_to_mask=True)
        elif args.kernel == 'alekernel':
            from nimare.meta.kernel import ALEKernel
            print("Running alekernel")
            k = ALEKernel(fwhm=15)
        if args.atlas is not None:
            imgs = k.transform(dset, return_type='image')
        else:
            time_series = np.transpose(k.transform(dset, return_type='array'))

    elif args.fmri_data is not None:
        imgs = nib.load(args.fmri_data).get_fdata()

    if args.atlas is not None:
        if args.atlas == 'harvard-oxford':
            print("Parcellating using the Harvard Oxford Atlas")
            atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
            atlas_labels = atlas.labels[1:]
            atlas_shape = atlas.maps.shape
            atlas_affine = atlas.maps.affine
            atlas_data = atlas.maps.get_fdata()
        elif args.atlas == 'aal':
            print("Parcellating using the AAL Atlas")
            atlas = datasets.fetch_atlas_aal()
            atlas_labels = atlas.labels
            atlas_shape = nib.load(atlas.maps).shape
            atlas_affine = nib.load(atlas.maps).affine
            atlas_data = nib.load(atlas.maps).get_fdata()
        elif args.atlas == 'craddock-2012':
            print("Parcellating using the Craddock-2012 Atlas")
            atlas = datasets.fetch_atlas_craddock_2012()
        elif args.atlas == 'destrieux-2009':
            print("Parcellating using the Destrieux-2009 Atlas")
            atlas = datasets.fetch_atlas_destrieux_2009(lateralized=True)
            atlas_labels = atlas.labels[3:]
            atlas_shape = nib.load(atlas.maps).shape
            atlas_affine = nib.load(atlas.maps).affine
            atlas_data = nib.load(atlas.maps).get_fdata()
            print(len(np.unique(atlas_data)))
        elif args.atlas == 'msdl':
            print("Parcellating using the MSDL Atlas")
            atlas = datasets.fetch_atlas_msdl()

        from nilearn.input_data import NiftiLabelsMasker
        masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True,
                                   memory='nilearn_cache')
        time_series = masker.fit_transform(imgs)

        #from nilearn.connectome import ConnectivityMeasure
        #print("Calculating correlation matrix")
        #correlation_measure = ConnectivityMeasure(kind='correlation')
        #correlation_matrix = correlation_measure.fit_transform([time_series])[0]

        # Mask the main diagonal for visualization:
        #plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=atlas_labels,
        #                    reorder=True)
        #plt.savefig(op.join(workdir, 'correlation_matrix.png'))

    print('Performing gradient analysis')

    optimal_grad_all = []
    lambdas_all = []
    #grads_all = []

    affinity_name = 'affinity-{0}'.format(args.affinity_kernel)
    approach_name = 'approach-{0}'.format(args.approach)
    colormap = np.transpose(np.vstack((np.linspace(0,1,10, endpoint=False), np.linspace(1,0,10, endpoint=False), np.linspace(0,1,10, endpoint=False))))
    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.set_xlabel('Component Nb')
    ax.set_ylabel('Eigenvalue')

    if args.atlas is not None:
        for i, tmp_thresh in enumerate(range(0,10,1)):
            tmp_thresh = tmp_thresh/10
            gm = GradientMaps(n_components=10, random_state=0, kernel=args.affinity_kernel, approach=args.approach)
            #gm.fit(correlation_matrix, sparsity=tmp_thresh, n_iter=100)
            gm.fit(time_series, sparsity=tmp_thresh, n_iter=100)
            ax.scatter(range(1,gm.lambdas_.size+1,1), gm.lambdas_, c=colormap[i,:], label='sparsity: {0}'.format(tmp_thresh))

            gm_lambdas_diff = gm.lambdas_[:-1] - gm.lambdas_[1:]

            optimal_num_gradients = np.where(gm_lambdas_diff == np.max(gm_lambdas_diff))[0][0] + 1

            lambdas_all.append(gm.lambdas_)
            optimal_grad_all.append(optimal_num_gradients)
            #grads_all.append(gm.gradients_)

            print('Optimal number of gradients is {0} for sparsity {1}'.format(optimal_num_gradients, tmp_thresh))

    else:
        gm = GradientMaps(n_components=10, random_state=0, kernel=args.affinity_kernel, approach=args.approach)
        #gm.fit(correlation_matrix, sparsity=tmp_thresh, n_iter=100)
        gm.fit(time_series, sparsity=None, n_iter=100)
        ax.scatter(range(1,gm.lambdas_.size+1,1), gm.lambdas_)

        gm_lambdas_diff = gm.lambdas_[:-1] - gm.lambdas_[1:]

        optimal_num_gradients = np.where(gm_lambdas_diff == np.max(gm_lambdas_diff))[0][0] + 1

        lambdas_all.append(gm.lambdas_)
        optimal_grad_all.append(optimal_num_gradients)

        print('Optimal number of gradients is {0}'.format(optimal_num_gradients))

    ax.legend()
    plt.savefig(op.join(workdir, 'lambdas.png'))

    lambdas_all = np.asarray(lambdas_all)

    from scipy import stats
    mode_optimal_num_gradients = stats.mode(optimal_grad_all).mode[0]
    gradients_name = 'gradients-{0}'.format(mode_optimal_num_gradients)
    print('The most consistent number of optimal gradients is {0} across all sparsity assesments'.format(mode_optimal_num_gradients))

    if args.atlas is not None:
        maxvar_sparsity = np.where(lambdas_all[:,mode_optimal_num_gradients-1] == np.max(lambdas_all[:,mode_optimal_num_gradients-1]))[0][0]
        sparsity_name = 'sparsity-{0}'.format(maxvar_sparsity/10)
        print('Optimal sparsity is {0}'.format(maxvar_sparsity/10))
    else:
        sparsity_name = 'sparsity-None'


    #optimal_grads = grads_all[maxvar_sparsity]
    # map the gradient to the parcels
    gm = GradientMaps(n_components=mode_optimal_num_gradients, random_state=0, kernel=args.affinity_kernel, approach=args.approach)
    #gm.fit(correlation_matrix, sparsity=maxvar_sparsity/10, n_iter=100)
    if args.atlas is not None:
        gm.fit(time_series, sparsity=maxvar_sparsity/10, n_iter=100)
    else:
        gm.fit(time_series, sparsity=None, n_iter=100)
    for i in range(mode_optimal_num_gradients):
        if args.atlas is not None:
            tmpimg = np.zeros(atlas_shape)
            for j, n in enumerate(np.unique(atlas_data)[1:]):
                inds = atlas_data == n
                #tmpimg[inds] = optimal_grads[j,i]
                tmpimg[inds] = gm.gradients_[j,i]
                nib.save(nib.Nifti1Image(tmpimg, atlas_affine), op.join(workdir, 'gradient-{0}.nii.gz'.format(i)))
        else:
            tmpimg = np.zeros(np.prod(dset.masker.mask_img.shape))
            inds = np.ravel_multi_index(np.nonzero(dset.masker.mask_img.get_fdata()), dset.masker.mask_img.shape)
            tmpimg[inds] = gm.gradients_[:,i]
            nib.save(nib.Nifti1Image(np.reshape(tmpimg, dset.masker.mask_img.shape), dset.masker.mask_img.affine), op.join(workdir, 'gradient-{0}.nii.gz'.format(i)))

        os.system('python3 /Users/miriedel/Desktop/GitHub/surflay/make_figures.py '
                  '-f {grad_image} --colormap jet'.format(grad_image = op.join(workdir, 'gradient-{0}.nii.gz'.format(i))))

    output_dir = op.join(args.outdir, '{dataset_name}_{atlas_name}_{kernel_name}_{sparsity_name}_{gradients_name}_{affinity_name}_{approach_name}'.format(
                                        dataset_name=dataset_name,
                                        atlas_name=atlas_name,
                                        kernel_name=kernel_name,
                                        sparsity_name=sparsity_name,
                                        gradients_name=gradients_name,
                                        affinity_name=affinity_name,
                                        approach_name=approach_name))
    os.makedirs(output_dir, exist_ok=True)
    os.rename(workdir, output_dir)


if __name__ == '__main__':
    main()
