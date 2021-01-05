"""
Based on
https://github.com/BIDS-Apps/example/blob/aa0d4808974d79c9fbe54d56d3b47bb2cf4e0a0d/run.py
"""
import os
import os.path as op
import argparse
import nibabel as nib
import numpy as np
import shutil
from nilearn import plotting
from nilearn import datasets
from nimare.io import convert_sleuth_to_dataset
from nilearn.input_data import NiftiLabelsMasker
from nimare.dataset import Dataset
from nimare.meta.kernel import ALEKernel
from nilearn.connectome import ConnectivityMeasure
import pickle
from nilearn import surface
from nilearn.datasets import fetch_surf_fsaverage
from mapalign import embed
from nilearn import masking
import utils
from glob import glob


def get_parser():
    parser = argparse.ArgumentParser(description='This script will generate axials, surface medial and surface lateral view images with the specified overlay.')
    parser.add_argument('--neurosynth', required=False, dest='neurosynth', action='store_true',
                        help=('Query the Neurosynth database.'))
    parser.add_argument('--subcortical', required=False, dest='subcort', action='store_true', default=False,
                        help=('Whether to include the subcortical voxels.'))
    parser.add_argument('--nimare-dataset', required=False, dest='nimare_dataset', default=None,
                        help=('Import a NiMARE dataset.'))
    parser.add_argument('--neurosynth-file', required=False, dest='neurosynth_file',
                        help='Full path to neurosynth file to use as database.')
    parser.add_argument('--sleuth-file', required=False, dest='sleuth_file',
                        help='Full path to sleuth file to use as database.')
    parser.add_argument('--roi-mask', required=False, dest='roi_mask',
                        help='Full path to roi mask for selecting studies.')
    parser.add_argument('--approach', required=False, dest='approach', default='dm',
                        help='Embedding approach for gradients.')
    parser.add_argument('--affinity', required=False, dest='affinity', default='cosine',
                        help='Kernel function to build the affinity matrix.')
    parser.add_argument('--term', required=False, dest='term',
                        help='Term or list of terms (e.g. [\'load\', \'rest\'] for selecting studies.')
    parser.add_argument('--topic', required=False, dest='topic', nargs='*',
                        help='Topic or list of topics (e.g. [\'topic002\', \'topic023\'] for selecting studies.')
    parser.add_argument('--kernel', required=False, dest='kernel', default='alekernel',
                        help='Kernel for converting peaks.')
    parser.add_argument('--atlas', required=False, dest='atlas', default='fsaverage5',
                        help='Atlas name for parcellating data: harvard-oxford, aal, craddock-2012, destrieux-2009, msdl, fsaverage5 (surface), hcp (surface)')
    parser.add_argument('--gradients', required=False, dest='gradients', default=None,
                        help='Number of gradients to produce.')
    parser.add_argument('--sparsity', required=False, dest='sparsity', default=0.9,
                        help='Sparsity for thresholding connectivity matrix.')
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
    sparsity_name = 'sparsity-{0}'.format(args.sparsity)
    affinity_name = 'affinity-{0}'.format(args.affinity)
    approach_name = 'approach-{0}'.format(args.approach)
    gradients_name = 'gradients-{0}'.format(args.gradients)
    subcortical_name='subcortical-{0}'.format(args.subcort)

    dset=None

    #handle neurosynth dataset, if called
    if args.neurosynth == True:
        if args.neurosynth_file is None:

            ns_data_dir = op.join(workdir, 'neurosynth')
            dataset_file = op.join(ns_data_dir, 'neurosynth_dataset.pkl.gz')
            # download neurosynth dataset if necessary
            if not op.isfile(dataset_file):
                utils.neurosynth_download(ns_data_dir)

        else:
            dataset_file = args.neurosynth_file

        dset = Dataset.load(dataset_file)
        dataset_name = 'dataset-neurosynth'

    #handle sleuth text file, if called
    if args.sleuth_file is not None:
        dset = convert_sleuth_to_dataset(args.sleuth_file, target="mni152_2mm")
        dataset_name = 'dataset-{0}'.format(op.basename(args.sleuth_file).split('.')[0])

    if args.nimare_dataset is not None:
        dset = Dataset.load(args.nimare_dataset)
        dataset_name = 'dataset-{0}'.format(op.basename(args.nimare_dataset).split('.')[0])

    if dset:
        #slice studies, if needed
        if args.roi_mask is not None:
            roi_ids = dset.get_studies_by_mask(args.roi_mask)
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write('{}/{} studies report at least one coordinate in the '
                    'ROI\n'.format(len(roi_ids), len(dset.ids)))
            dset_sel = dset.slice(roi_ids)
            dset = dset_sel
            dataset_name = 'dataset-neurosynth_mask-{0}'.format(op.basename(args.roi_mask).split('.')[0])

        if args.term is not None:
            labels = ['Neurosynth_TFIDF__{label}'.format(label=label) for label in [args.term]]
            term_ids = dset.get_studies_by_label(labels=labels, label_threshold=0.1)
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write('{}/{} studies report association '
                    'with the term {}\n'.format(len(term_ids), len(dset.ids), args.term))
            dset_sel = dset.slice(term_ids)
            dset = dset_sel
            img_inds = np.nonzero(dset.masker.mask_img.get_fdata())
            vox_locs = np.unravel_index(img_inds, dset.masker.mask_img.shape)
            dataset_name = 'dataset-neurosynth_term-{0}'.format(args.term)

        if args.topic is not None:
            topics = ['Neurosynth_{version}__{topic}'.format(version=args.topic[0], topic=topic) for topic in args.topic[1:]]
            topics_ids = []
            for topic in topics:
                topic_ids = dset.annotations.id[np.where(dset.annotations[topic])[0]].tolist()
                topics_ids.extend(topic_ids)
                with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                    fo.write('{}/{} studies report association '
                        'with the term {}\n'.format(len(topic_ids), len(dset.ids), topic))
            topics_ids_unique = np.unique(topics_ids)
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write('{} unique ids\n'.format(len(topics_ids_unique)))
            dset_sel = dset.slice(topics_ids_unique)
            dset = dset_sel
            img_inds = np.nonzero(dset.masker.mask_img.get_fdata())
            vox_locs = np.unravel_index(img_inds, dset.masker.mask_img.shape)
            dataset_name = 'dataset-neurosynth_topic-{0}'.format('_'.join(args.topic[1:]))

        if (args.neurosynth == True) or (args.sleuth_file is not None) or (args.nimare_dataset is not None):
            if args.kernel == 'peaks2maps':
                with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                    fo.write("Running peak2maps\n")
                k = Peaks2MapsKernel(resample_to_mask=True)
            elif args.kernel == 'alekernel':
                with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                    fo.write("Running alekernel\n")
                k = ALEKernel(fwhm=15)
            if args.atlas is not None:
                imgs = k.transform(dset, return_type='image')
            else:
                time_series = np.transpose(k.transform(dset, return_type='array'))

    if args.atlas is not None:
        if args.atlas == 'harvard-oxford':
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write("Parcellating using the Harvard Oxford Atlas\n")
            atlas_labels = atlas.labels[1:]
            atlas_shape = atlas.maps.shape
            atlas_affine = atlas.maps.affine
            atlas_data = atlas.maps.get_fdata()
        elif args.atlas == 'aal':
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write("Parcellating using the AAL Atlas\n")
            atlas = datasets.fetch_atlas_aal()
            atlas_labels = atlas.labels
            atlas_shape = nib.load(atlas.maps).shape
            atlas_affine = nib.load(atlas.maps).affine
            atlas_data = nib.load(atlas.maps).get_fdata()
        elif args.atlas == 'craddock-2012':
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write("Parcellating using the Craddock-2012 Atlas\n")
            atlas = datasets.fetch_atlas_craddock_2012()
        elif args.atlas == 'destrieux-2009':
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write("Parcellating using the Destrieux-2009 Atlas\n")
            atlas = datasets.fetch_atlas_destrieux_2009(lateralized=True)
            atlas_labels = atlas.labels[3:]
            atlas_shape = nib.load(atlas.maps).shape
            atlas_affine = nib.load(atlas.maps).affine
            atlas_data = nib.load(atlas.maps).get_fdata()
        elif args.atlas == 'msdl':
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write("Parcellating using the MSDL Atlas\n")
            atlas = datasets.fetch_atlas_msdl()
        elif args.atlas == 'surface':
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write("Generating surface vertices\n")

        if args.atlas != "fsaverage5" and args.atlas != 'hcp' :
            masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True,
                                       memory='nilearn_cache')
            time_series = masker.fit_transform(imgs)

        else:
            if args.atlas == 'fsaverage5':
                fsaverage = fetch_surf_fsaverage(mesh='fsaverage5')
                pial_left = fsaverage.pial_left
                pial_right = fsaverage.pial_right
                medial_wall_inds_left = surface.load_surf_data('./templates/lh.Medial_wall.label')
                medial_wall_inds_right = surface.load_surf_data('./templates/rh.Medial_wall.label')
                sulc_left = fsaverage.sulc_left
                sulc_right = fsaverage.sulc_right

            elif args.atlas == 'hcp':
                pial_left = './templates/S1200.L.pial_MSMAll.32k_fs_LR.surf.gii'
                pial_right = './templates/S1200.R.pial_MSMAll.32k_fs_LR.surf.gii'
                medial_wall_inds_left = np.where(nib.load('./templates/hcp.tmp.lh.dscalar.nii').get_fdata()[0] == 0)[0]
                medial_wall_inds_right = np.where(nib.load('./templates/hcp.tmp.rh.dscalar.nii').get_fdata()[0] == 0)[0]
                left_verts = 32492-len(medial_wall_inds_left)
                sulc_left = nib.load('./templates/S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii').get_fdata()[0][0:left_verts]*-1
                sulc_left = utils.insert(sulc_left, medial_wall_inds_left)
                sulc_right = nib.load('./templates/S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii').get_fdata()[0][left_verts:]*-1
                sulc_right = utils.insert(sulc_right, medial_wall_inds_right)

            surf_lh = surface.vol_to_surf(imgs, pial_left, radius=6.0, interpolation='nearest', kind='ball', n_samples=None, mask_img=dset.masker.mask_img)
            surf_rh = surface.vol_to_surf(imgs, pial_right, radius=6.0, interpolation='nearest', kind='ball', n_samples=None, mask_img=dset.masker.mask_img)
            lh_vertices_total = np.shape(surf_lh)[0]
            rh_vertices_total = np.shape(surf_rh)[0]
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write('{0} vertices in left hemisphere after conversion to {1} surface space\n'.format(lh_vertices_total, args.atlas))
                fo.write('{0} vertices in right hemisphere after conversion to {1} surface space\n'.format(rh_vertices_total, args.atlas))

            #calculate an ALE image of studies
            surf_ale_array_lh = 1.0 - np.prod(1.0 - surf_lh, axis=1)
            surf_ale_array_rh = 1.0 - np.prod(1.0 - surf_rh, axis=1)

            #create dictionary for plotting (should change key for [0] and [1])
            ale_dict = {'grads_lh': np.expand_dims(surf_ale_array_lh, axis=1),
                         'grads_rh': np.expand_dims(surf_ale_array_rh, axis=1),
                         'pial_left': pial_left,
                         'sulc_left': sulc_left,
                         'pial_right': pial_right,
                         'sulc_right': sulc_right}

            del surf_ale_array_lh, surf_ale_array_rh

            utils.plot_surfaces(ale_dict, 0, workdir, 'ale', normalize=False, cmap='nipy_spectral')
            im_list = [op.join(workdir, 'ale-0_left_lateral.png'),
                       op.join(workdir, 'ale-0_left_medial.png'),
                       op.join(workdir, 'ale-0_right_medial.png'),
                       op.join(workdir, 'ale-0_right_lateral.png')]
            utils.combine_plots(im_list, op.join(workdir, 'ale.png') )

            if args.subcort:
                surf_lh = np.delete(surf_lh, medial_wall_inds_left, axis=0)
                surf_rh = np.delete(surf_rh, medial_wall_inds_right, axis=0)
                lh_vertices_wo_medial_wall = np.shape(surf_lh)[0]
                rh_vertices_wo_medial_wall = np.shape(surf_rh)[0]
                with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                    fo.write('{0} vertices in left hemisphere after removing {1} left medial wall vertices\n'.format(lh_vertices_wo_medial_wall, len(medial_wall_inds_left)))
                    fo.write('{0} vertices in right hemisphere after removing {1} right medial wall vertices\n'.format(rh_vertices_wo_medial_wall, len(medial_wall_inds_right)))

                with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                    fo.write('adding subcortical voxels\n')
                subcort_img = nib.load('templates/rois-subcortical_mni152_mask.nii.gz')
                subcort_ts = masking.apply_mask(imgs, mask_img=subcort_img)
                num_subcort_vox = np.shape(subcort_ts)[1]
                with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                    fo.write('Adding time-series for {} sub-cortical voxels\n'.format(num_subcort_vox))
                time_series = np.hstack((np.transpose(np.vstack((surf_lh, surf_rh))), subcort_ts))

                del subcort_ts

            else:
                time_series = np.transpose(np.vstack((surf_lh, surf_rh)))

            del imgs, surf_lh, surf_rh

        with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
            fo.write('Matrix contains {0} voxels/vertices across {1} MA images\n'.format(np.shape(time_series)[1], np.shape(time_series)[0]))
        time_series = time_series.astype('float32')

        inds_discard = np.append(np.where(np.isnan(np.mean(time_series, axis=0)) == True)[0], np.where(np.any(time_series, axis=0) == False)[0])
        if inds_discard.any():
            time_series = np.delete(time_series, inds_discard, axis=1)
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write('removing {} vertices and voxels without MA values\n'.format(len(inds_discard)))
                fo.write('Matrix contains {0} voxels/vertices across {1} MA images\n'.format(np.shape(time_series)[1], np.shape(time_series)[0]))

        with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
            fo.write('calculating correlation matrix\n')
        correlation = ConnectivityMeasure(kind='correlation')
        time_series = correlation.fit_transform([time_series])[0]

        if args.affinity == "cosine":
            with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
                fo.write('calculating affinity matrix\n')
            time_series = utils.affinity(time_series, 10*args.sparsity)
            with open(op.join(workdir, 'affinity-matrix.p'), 'wb') as fo:
                pickle.dump(time_series, fo, protocol=4)

    with open(op.join(workdir, 'analysis-information.txt'), 'a+') as fo:
        fo.write('Performing gradient analysis\n')

    gradients, statistics = embed.compute_diffusion_map(time_series, alpha = 0.5, return_result=True, overwrite=True)
    pickle.dump(statistics, open(op.join(workdir, 'statistics.p'), "wb"))

    # putting vertices w/o time-series information back in gradients with value=0
    if inds_discard.any():
        gradients = utils.insert(gradients, inds_discard)

    # if subcortical included in gradient decomposition, remove gradient scores
    if args.subcort:
        subcort_grads = gradients[np.shape(gradients)[0]-num_subcort_vox:,:]
        gradients = gradients[0:np.shape(gradients)[0]-num_subcort_vox,:]

        # get left hemisphere gradient scores, and insert 0's where medial wall is
        gradients_lh = gradients[0:lh_vertices_wo_medial_wall,:]
        gradients_lh = utils.insert(gradients_lh, medial_wall_inds_left)

        # get right hemisphere gradient scores and insert 0's where medial wall is
        gradients_rh = gradients[-rh_vertices_wo_medial_wall:,:]
        gradients_rh = utils.insert(gradients_rh, medial_wall_inds_right)

    else:
        gradients_lh = gradients[0:int(np.shape(gradients)[0]/2),:]
        gradients_rh = gradients[int(np.shape(gradients)[0]/2):,:]

    grad_dict = {'grads_lh': gradients_lh,
                 'grads_rh': gradients_rh,
                 'pial_left': pial_left,
                 'sulc_left': sulc_left,
                 'pial_right': pial_right,
                 'sulc_right': sulc_right}
    if args.subcort:
        grad_dict['subcort_grads'] = subcort_grads
    pickle.dump(grad_dict, open(op.join(workdir, 'gradients.p'), "wb"))

    #find the number of components that explain at least 50% variance
    n_components = np.where(np.cumsum(statistics['lambdas']/np.sum(statistics['lambdas'])) > 0.5)[0][0]+1

    # map the gradient to the parcels
    for i in range(n_components):
        if args.atlas is not None:
            if args.atlas == 'fsaverage5' or args.atlas =='hcp':

                utils.plot_surfaces(grad_dict, i, workdir, 'gradient', normalize=False)
                im_list = [op.join(workdir, 'gradient-{}_left_lateral.png'.format(i)),
                           op.join(workdir, 'gradient-{}_left_medial.png'.format(i)),
                           op.join(workdir, 'gradient-{}_right_medial.png'.format(i)),
                           op.join(workdir, 'gradient-{}_right_lateral.png'.format(i))]
                utils.combine_plots(im_list, op.join(workdir, 'gradient-{0}.png'.format(i)))

                if args.subcort:
                    tmpimg = masking.unmask(subcort_grads[:,i], subcort_img)
                    nib.save(tmpimg, op.join(workdir, 'gradient-{0}.nii.gz'.format(i)))
            else:
                tmpimg = np.zeros(atlas_shape)
                for j, n in enumerate(np.unique(atlas_data)[1:]):
                    inds = atlas_data == n
                    tmpimg[inds] = gradients[j,i]
                    nib.save(nib.Nifti1Image(tmpimg, atlas_affine), op.join(workdir, 'gradient-{0}.nii.gz'.format(i)))
        else:
            tmpimg = np.zeros(np.prod(dset.masker.mask_img.shape))
            inds = np.ravel_multi_index(np.nonzero(dset.masker.mask_img.get_fdata()), dset.masker.mask_img.shape)
            tmpimg[inds] = gradients[:,i]
            nib.save(nib.Nifti1Image(np.reshape(tmpimg, dset.masker.mask_img.shape), dset.masker.mask_img.affine), op.join(workdir, 'gradient-{0}.nii.gz'.format(i)))
            #include a command for surface plots, if desired


    output_dir = op.join(args.outdir, '{dataset_name}_{atlas_name}_{kernel_name}_{sparsity_name}_{gradients_name}_{affinity_name}_{approach_name}_{subcortical_name}'.format(
                                        dataset_name=dataset_name,
                                        atlas_name=atlas_name,
                                        kernel_name=kernel_name,
                                        sparsity_name=sparsity_name,
                                        gradients_name=gradients_name,
                                        affinity_name=affinity_name,
                                        approach_name=approach_name,
                                        subcortical_name=subcortical_name))

    shutil.copytree(workdir, output_dir)

    shutil.rmtree(workdir)


if __name__ == '__main__':
    main()
