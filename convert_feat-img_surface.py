import os
import os.path as op
from glob import glob
import nibabel as nib
import numpy as np
from nilearn import surface
from nilearn.masking import apply_mask
from nilearn.datasets import load_mni152_brain_mask
from nilearn.datasets import fetch_surf_fsaverage
import pickle


image_dir = '/home/data/nbc/misc-projects/meta-gradients/code/feature_maps'
out_dir = '/home/data/nbc/misc-projects/meta-gradients/code'

mask = load_mni152_brain_mask()
feature_images = sorted(glob(op.join(image_dir, '*_association-test_z.nii.gz')))

surf_dict = {}

for img in feature_images:

    tmp_feat_name = op.basename(img).split('_')[0]
    print(tmp_feat_name)
    fsaverage = fetch_surf_fsaverage(mesh='fsaverage5')
    surf_lh = surface.vol_to_surf(nib.load(img), fsaverage.pial_left, radius=6.0, interpolation='nearest', kind='ball', n_samples=None, mask_img=mask)
    surf_rh = surface.vol_to_surf(nib.load(img), fsaverage.pial_right, radius=6.0, interpolation='nearest', kind='ball', n_samples=None, mask_img=mask)
    surf = np.transpose(np.hstack((surf_lh, surf_rh)))

    surf_dict[tmp_feat_name] = surf

with open(op.join(out_dir, 'ns_feature_maps_surface.pkl.gz'), 'wb') as fo:
    pickle.dump(surf_dict, fo)
