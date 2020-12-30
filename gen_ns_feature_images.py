from neurosynth.base.dataset import Dataset
from neurosynth.analysis import meta
import os
import os.path as op
import pickle


""" Create a new Dataset instance from a database file and load features.
This is basically the example from the quickstart in the README.
Assumes you have database.txt and features.txt files in the current dir.
"""

""" Load a Dataset and generate a full set of meta-analysis
images--i.e., run a meta-analysis on every single feature.
"""

neurosynth_data_dir = '/home/data/nbc/misc-projects/niconn-macm/code/neurosynth/'

if not op.isfile(op.join(neurosynth_data_dir, 'dataset.pkl')):
    # Create Dataset instance from a database file.
    dataset = Dataset(op.join(neurosynth_data_dir, 'database.txt'))

    # Load features from file
    dataset.add_features(op.join(neurosynth_data_dir, 'features.txt'))

    # Pickle the Dataset to file so we can use Dataset.load() next time
    # instead of having to sit through the generation process again.
    dataset.save(op.join(neurosynth_data_dir, 'dataset.pkl'))

# Load pickled Dataset--assumes you've previously saved it. If not,
# follow the create_a_new_dataset_and_load_features example.
dataset = Dataset.load(op.join(neurosynth_data_dir, 'dataset.pkl'))

# Get the full list of feature names
feature_list = dataset.get_feature_names()

# Run a meta-analysis on each feature, and save all the results to
# a directory called results. Note that the directory will not be
# created for you, so make sure it exists.
# Here we use the default frequency threshold of 0.001 (i.e., a
# study is said to have a feature if more than 1 in every 1,000
# words is the target word), and an FDR correction level of 0.05.
out_dir = '/home/data/nbc/misc-projects/meta-gradients/code/feature_maps'

for tmp_feature in feature_list:
    print(tmp_feature)
    meta.analyze_features(dataset, [tmp_feature], threshold=0.001, image_type='association-test_z', output_dir=out_dir, q=0.01)
