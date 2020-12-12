"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))  # append the MIALab root directory to Python path
# fixes the ModuleNotFoundError when executing main.py in the console after code changes (e.g. git pull)
# somehow pip install does not keep track of packages

import mialab.data.structure as structure
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform,
                structure.BrainImageTypes.RegistrationParameterMap]  # the list of data we will load


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    np.random.seed(42)  # set fixed seed

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    pre_processed = True
    is_non_rigid = True

    atlas_based_seg = True
    shaped_based_averaging = True
    weighted_atlas = False
    local_weights = False

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter(),
                                          pre_processed=pre_processed,
                                          is_non_rigid=is_non_rigid)

    if atlas_based_seg and pre_processed:
        pre_process_params = None
    else:
        pre_process_params = {'skullstrip_pre': not pre_processed,
                              'normalization_pre': not pre_processed,
                              'registration_pre': not pre_processed,
                              'non_rigid_registration': is_non_rigid,
                              'coordinates_feature': not atlas_based_seg,
                              'intensity_feature': not atlas_based_seg,
                              'gradient_intensity_feature': not atlas_based_seg}

    # load images for training and pre-process
    if atlas_based_seg:
        # Load atlas files
        if weighted_atlas:
            images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)
        if shaped_based_averaging:
            if is_non_rigid:
                predictions = np.load(os.path.join(data_atlas_dir, 'atlas_prediction_SBA_non_rigid.npy'))
                probabilities = predictions
            else:
                predictions = np.load(os.path.join(data_atlas_dir, 'atlas_prediction_SBA_affine.npy'))
                probabilities = predictions
        elif is_non_rigid:
            predictions = np.load(os.path.join(data_atlas_dir, 'atlas_prediction_non_rigid.npy'))
            probabilities = np.load(os.path.join(data_atlas_dir, 'atlas_probabilities_non_rigid.npy'))
        else:
            predictions = np.load(os.path.join(data_atlas_dir, 'atlas_prediction_affine.npy'))
            probabilities = np.load(os.path.join(data_atlas_dir, 'atlas_probabilities_affine.npy'))
    else:
        print('-' * 5, 'Training...')
        images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)
        # generate feature matrix and label vector
        data_train = np.concatenate([img.feature_matrix[0] for img in images])
        labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

        forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                    n_estimators=10,
                                                    max_depth=40)
        start_time = timeit.default_timer()
        forest.fit(data_train, labels_train)
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    # Create a atlas with the GroundTruth
    # putil.create_atlas(images,is_non_rigid)
    # putil.create_sba_atlas(images, is_non_rigid)

    # Load atlas files
    # putil.display_slice(atlas_prediction, 100)

    # generate feature matrix and label vector
    # data_train = np.concatenate([img.feature_matrix[0] for img in images])
    # labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    # forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
    #                                           n_estimators=10,
    #                                           max_depth=10)

    start_time = timeit.default_timer()
    # forest.fit(data_train, labels_train)
    print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator()

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter(),
                                          pre_processed=pre_processed,
                                          is_non_rigid=is_non_rigid)

    # load images for testing and pre-process
    if not atlas_based_seg:
        pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # putil.display_slice([img.images[structure.BrainImageTypes.GroundTruth] for img in images_test], 100)

    images_prediction = []
    images_probabilities = []

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)
        if atlas_based_seg:
            if weighted_atlas:
                if local_weights:
                    predictions, probabilities = putil.local_weighted_atlas(img, images)
                else:
                    predictions, probabilities = putil.global_weighted_atlas(img, images)
        else:
            start_time = timeit.default_timer()
            predictions = forest.predict(img.feature_matrix[0])
            probabilities = forest.predict_proba(img.feature_matrix[0])
            print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        start_time = timeit.default_timer()

        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions,
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # post-process segmentation and evaluate with post-processing
    post_process_params = {'simple_post': True}
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                     post_process_params, multi_process=False)



    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                           img.id_ + '-PP')

        # save results
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
        sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    result_file = os.path.join(result_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()

if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
