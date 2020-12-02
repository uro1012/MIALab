"""This module contains utility classes and functions."""
import enum
import os
import typing as t
import warnings

import numpy as np
from scipy import stats
from scipy import special
from sklearn.utils.extmath import weighted_mode
import pymia.data.conversion as conversion
import pymia.filtering.filter as fltr
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric
import SimpleITK as sitk
from matplotlib import pyplot as plt

import mialab.data.structure as structure
import mialab.filtering.feature_extraction as fltr_feat
import mialab.filtering.postprocessing as fltr_postp
import mialab.filtering.preprocessing as fltr_prep
import mialab.utilities.multi_processor as mproc

atlas_t1 = sitk.Image()
atlas_t2 = sitk.Image()


def load_atlas_images(directory: str):
    """Loads the T1 and T2 atlas images.

    Args:
        directory (str): The atlas data directory.
    """

    global atlas_t1
    global atlas_t2
    atlas_t1 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz'))
    atlas_t2 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t2_tal_nlin_sym_09a.nii.gz'))
    if not conversion.ImageProperties(atlas_t1) == conversion.ImageProperties(atlas_t2):
        raise ValueError('T1w and T2w atlas images have not the same image properties')


class FeatureImageTypes(enum.Enum):
    """Represents the feature image types."""

    ATLAS_COORD = 1
    T1w_INTENSITY = 2
    T1w_GRADIENT_INTENSITY = 3
    T2w_INTENSITY = 4
    T2w_GRADIENT_INTENSITY = 5


class FeatureExtractor:
    """Represents a feature extractor."""

    def __init__(self, img: structure.BrainImage, **kwargs):
        """Initializes a new instance of the FeatureExtractor class.

        Args:
            img (structure.BrainImage): The image to extract features from.
        """
        self.img = img
        self.training = kwargs.get('training', True)
        self.coordinates_feature = kwargs.get('coordinates_feature', False)
        self.intensity_feature = kwargs.get('intensity_feature', False)
        self.gradient_intensity_feature = kwargs.get('gradient_intensity_feature', False)

    def execute(self) -> structure.BrainImage:
        """Extracts features from an image.

        Returns:
            structure.BrainImage: The image with extracted features.
        """
        # warnings.warn('No features from T2-weighted image extracted.')
        generateFeatureMatrix = False

        if self.coordinates_feature:
            atlas_coordinates = fltr_feat.AtlasCoordinates()
            self.img.feature_images[FeatureImageTypes.ATLAS_COORD] = \
                atlas_coordinates.execute(self.img.images[structure.BrainImageTypes.T1w])
            generateFeatureMatrix = True

        if self.intensity_feature:
            self.img.feature_images[FeatureImageTypes.T1w_INTENSITY] = self.img.images[structure.BrainImageTypes.T1w]
            self.img.feature_images[FeatureImageTypes.T2w_INTENSITY] = self.img.images[structure.BrainImageTypes.T2w]
            generateFeatureMatrix = True

        if self.gradient_intensity_feature:
            # compute gradient magnitude images
            self.img.feature_images[FeatureImageTypes.T1w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T1w])
            self.img.feature_images[FeatureImageTypes.T2w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T2w])
            generateFeatureMatrix = True

        if generateFeatureMatrix:
            self._generate_feature_matrix()

        return self.img

    def _generate_feature_matrix(self):
        """Generates a feature matrix."""

        mask = None
        if self.training:
            # generate a randomized mask where 1 represents voxels used for training
            # the mask needs to be binary, where the value 1 is considered as a voxel which is to be loaded
            # we have following labels:
            # - 0 (background)
            # - 1 (white matter)
            # - 2 (grey matter)
            # - 3 (Hippocampus)
            # - 4 (Amygdala)
            # - 5 (Thalamus)

            # you can exclude background voxels from the training mask generation
            # mask_background = self.img.images[structure.BrainImageTypes.BrainMask]
            # and use background_mask=mask_background in get_mask()

            mask = fltr_feat.RandomizedTrainingMaskGenerator.get_mask(
                self.img.images[structure.BrainImageTypes.GroundTruth],
                [0, 1, 2, 3, 4, 5],
                [0.0003, 0.004, 0.003, 0.04, 0.04, 0.02])

            # convert the mask to a logical array where value 1 is False and value 0 is True
            mask = sitk.GetArrayFromImage(mask)
            mask = np.logical_not(mask)

        # generate features
        data = np.concatenate(
            [self._image_as_numpy_array(image, mask) for id_, image in self.img.feature_images.items()],
            axis=1)

        # generate labels (note that we assume to have a ground truth even for testing)
        labels = self._image_as_numpy_array(self.img.images[structure.BrainImageTypes.GroundTruth], mask)

        self.img.feature_matrix = (data.astype(np.float32), labels.astype(np.int16))

    @staticmethod
    def _image_as_numpy_array(image: sitk.Image, mask: np.ndarray = None):
        """Gets an image as numpy array where each row is a voxel and each column is a feature.

        Args:
            image (sitk.Image): The image.
            mask (np.ndarray): A mask defining which voxels to return. True is background, False is a masked voxel.

        Returns:
            np.ndarray: An array where each row is a voxel and each column is a feature.
        """

        number_of_components = image.GetNumberOfComponentsPerPixel()  # the number of features for this image
        no_voxels = np.prod(image.GetSize())
        image = sitk.GetArrayFromImage(image)

        if mask is not None:
            no_voxels = np.size(mask) - np.count_nonzero(mask)

            if number_of_components == 1:
                masked_image = np.ma.masked_array(image, mask=mask)
            else:
                # image is a vector image, make a vector mask
                vector_mask = np.expand_dims(mask, axis=3)  # shape is now (z, x, y, 1)
                vector_mask = np.repeat(vector_mask, number_of_components,
                                        axis=3)  # shape is now (z, x, y, number_of_components)
                masked_image = np.ma.masked_array(image, mask=vector_mask)

            image = masked_image[~masked_image.mask]

        return image.reshape((no_voxels, number_of_components))


def pre_process(id_: str, paths: dict, **kwargs) -> structure.BrainImage:
    """Loads and processes an image.

    The processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        id_ (str): An image identifier.
        paths (dict): A dict, where the keys are an image identifier of type structure.BrainImageTypes
            and the values are paths to the images.

    Returns:
        (structure.BrainImage):
    """

    print('-' * 10, 'Processing', id_)
    images_to_plot = []

    # load image
    path = paths.pop(id_, '')  # the value with key id_ is the root directory of the image
    path_to_transform = paths.pop(structure.BrainImageTypes.RegistrationTransform, '')
    path_to_parameterMap = paths.pop(structure.BrainImageTypes.RegistrationParameterMap, '')

    img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}

    if not kwargs:
        img = structure.BrainImage(id_, path, img, None, None)
        return img

    is_non_rigid = False
    if kwargs.get('non_rigid_registration', False):
        is_non_rigid = True
    transform = sitk.ReadTransform(path_to_transform)
    # parameterMap = findTransform(atlas_t1, sitk.Mask(img[structure.BrainImageTypes.T1w], img[structure.BrainImageTypes.BrainMask]))
    parameterMap = (sitk.ReadParameterFile(path_to_parameterMap + '_0.txt'),
                    sitk.ReadParameterFile(path_to_parameterMap + '_1.txt'))

    img = structure.BrainImage(id_, path, img, transform, parameterMap)



    if id_ == '100307':
        images_to_plot.append(img.images[structure.BrainImageTypes.T1w])

    # construct pipeline for brain mask registration
    # we need to perform this before the T1w and T2w pipeline because the registered mask is used for skull-stripping
    pipeline_brain_mask = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_brain_mask.add_filter(fltr_prep.ImageRegistration())
        pipeline_brain_mask.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, img.parameterMap, True, is_non_rigid),
                              len(pipeline_brain_mask.filters) - 1)

    # execute pipeline on the brain mask image
    img.images[structure.BrainImageTypes.BrainMask] = pipeline_brain_mask.execute(
        img.images[structure.BrainImageTypes.BrainMask])

    # construct pipeline for T1w image pre-processing
    pipeline_t1 = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageRegistration())
        pipeline_t1.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, img.parameterMap, is_non_rigid=is_non_rigid),
                              len(pipeline_t1.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t1.add_filter(fltr_prep.SkullStripping())
        pipeline_t1.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                              len(pipeline_t1.filters) - 1)
    if kwargs.get('normalization_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageNormalization())

    # execute pipeline on the T1w image
    img.images[structure.BrainImageTypes.T1w] = pipeline_t1.execute(img.images[structure.BrainImageTypes.T1w])

    # construct pipeline for T2w image pre-processing
    pipeline_t2 = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t2.add_filter(fltr_prep.ImageRegistration())
        pipeline_t2.set_param(fltr_prep.ImageRegistrationParameters(atlas_t2, img.transformation, img.parameterMap, is_non_rigid=is_non_rigid),
                              len(pipeline_t2.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t2.add_filter(fltr_prep.SkullStripping())
        pipeline_t2.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                              len(pipeline_t2.filters) - 1)
    if kwargs.get('normalization_pre', False):
        pipeline_t2.add_filter(fltr_prep.ImageNormalization())

    # images_to_plot.append(img.images[structure.BrainImageTypes.T1w])
    # images_to_plot.append(atlas_t1)
    # display_slice(images_to_plot, 100)

    # execute pipeline on the T2w image
    img.images[structure.BrainImageTypes.T2w] = pipeline_t2.execute(img.images[structure.BrainImageTypes.T2w])

    # construct pipeline for ground truth image pre-processing
    pipeline_gt = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_gt.add_filter(fltr_prep.ImageRegistration())
        pipeline_gt.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, img.parameterMap, True, is_non_rigid),
                              len(pipeline_gt.filters) - 1)

    # execute pipeline on the ground truth image
    img.images[structure.BrainImageTypes.GroundTruth] = pipeline_gt.execute(
        img.images[structure.BrainImageTypes.GroundTruth])

    # update image properties to atlas image properties after registration
    img.image_properties = conversion.ImageProperties(img.images[structure.BrainImageTypes.T1w])

    # extract the features
    feature_extractor = FeatureExtractor(img, **kwargs)
    img = feature_extractor.execute()

    img.feature_images = {}  # we free up memory because we only need the img.feature_matrix
    # for training of the classifier

    return img


def post_process(img: structure.BrainImage, segmentation: sitk.Image, probability: sitk.Image,
                 **kwargs) -> sitk.Image:
    """Post-processes a segmentation.

    Args:
        img (structure.BrainImage): The image.
        segmentation (sitk.Image): The segmentation (label image).
        probability (sitk.Image): The probabilities images (a vector image).

    Returns:
        sitk.Image: The post-processed image.
    """

    print('-' * 10, 'Post-processing', img.id_)

    # construct pipeline
    pipeline = fltr.FilterPipeline()
    if kwargs.get('simple_post', False):
        pipeline.add_filter(fltr_postp.ImagePostProcessing())
    if kwargs.get('crf_post', False):
        pipeline.add_filter(fltr_postp.DenseCRF())
        pipeline.set_param(fltr_postp.DenseCRFParams(img.images[structure.BrainImageTypes.T1w],
                                                     img.images[structure.BrainImageTypes.T2w],
                                                     probability), len(pipeline.filters) - 1)

    return pipeline.execute(segmentation)


def init_evaluator() -> eval_.Evaluator:
    """Initializes an evaluator.

    Returns:
        eval.Evaluator: An evaluator.
    """

    # initialize metrics
    metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(95.0)]
    # todo: add hausdorff distance, 95th percentile (see metric.HausdorffDistance)
    # warnings.warn('Initialized evaluation with the Dice coefficient. Do you know other suitable metrics?')

    # define the labels to evaluate
    labels = {1: 'WhiteMatter',
              2: 'GreyMatter',
              3: 'Hippocampus',
              4: 'Amygdala',
              5: 'Thalamus'
              }

    evaluator = eval_.SegmentationEvaluator(metrics, labels)
    return evaluator


def pre_process_batch(data_batch: t.Dict[structure.BrainImageTypes, structure.BrainImage],
                      pre_process_params: dict=None, multi_process=True) -> t.List[structure.BrainImage]:
    """Loads and pre-processes a batch of images.

    The pre-processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        data_batch (Dict[structure.BrainImageTypes, structure.BrainImage]): Batch of images to be processed.
        pre_process_params (dict): Pre-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.

    Returns:
        List[structure.BrainImage]: A list of images.
    """

    if pre_process_params is None:
        pre_process_params = {}


    params_list = list(data_batch.items())
    if multi_process:
        images = mproc.MultiProcessor.run(pre_process, params_list, pre_process_params, mproc.PreProcessingPickleHelper)
    else:
        images = [pre_process(id_, path, **pre_process_params) for id_, path in params_list]
    return images


def post_process_batch(brain_images: t.List[structure.BrainImage], segmentations: t.List[sitk.Image],
                       probabilities: t.List[sitk.Image], post_process_params: dict=None,
                       multi_process=True) -> t.List[sitk.Image]:
    """ Post-processes a batch of images.

    Args:
        brain_images (List[structure.BrainImageTypes]): Original images that were used for the prediction.
        segmentations (List[sitk.Image]): The predicted segmentation.
        probabilities (List[sitk.Image]): The prediction probabilities.
        post_process_params (dict): Post-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.

    Returns:
        List[sitk.Image]: List of post-processed images
    """
    if post_process_params is None:
        post_process_params = {}

    param_list = zip(brain_images, segmentations, probabilities)
    if multi_process:
        pp_images = mproc.MultiProcessor.run(post_process, param_list, post_process_params,
                                             mproc.PostProcessingPickleHelper)
    else:
        pp_images = [post_process(img, seg, prob, **post_process_params) for img, seg, prob in param_list]

    return pp_images


def findTransform(fixed, moving):
    moving = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkFloat32)

    # non-Rigid Registration elastix
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    transformixParameter = elastixImageFilter.GetTransformParameterMap()
    return transformixParameter


def display_slice(images, slice, seq_plot=False):
    if seq_plot:
        fig = plt.figure(figsize=(8, 8))

        n_row_plot = 2
        n_col_plot = 5
        for i in range(1, len(images)+1):
            fig.add_subplot(n_row_plot, n_col_plot, i)
            image_3d_np = sitk.GetArrayFromImage(images[i-1])
            image_2d_np = image_3d_np[slice, :, :]

            plt.imshow(image_2d_np, interpolation='nearest')
            plt.draw()
    else:
        image_3d_np1 = sitk.GetArrayFromImage(images[1])
        image_2d_np1 = image_3d_np1[slice, :, :]
        image_3d_np2 = sitk.GetArrayFromImage(images[2])
        image_2d_np2 = image_3d_np2[slice, :, :]

        image = np.zeros([233, 197, 3])
        image[:, :, 0] = image_2d_np1/4.0
        image[:, :, 1] = image_2d_np2/4.0
        plt.imshow(image)
    plt.show()


def create_atlas(images, isNonRigid):
    # Get the list of GroundTruth and converts the image in numpy format
    image_np = [sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.GroundTruth]) for img in images]

    # Stack the images in a 4-D numpy array
    images_np = np.stack(image_np,  axis=-1)

    # Compile the atlas by taking the most occurring label for each voxel
    atlas_np = stats.mode(images_np, axis=3)

    # Remove 4th dimension
    atlas_predictions = np.squeeze(atlas_np.mode)
    atlas_probabilities = np.squeeze(atlas_np.count)/len(images)

    # Save atlas
    if isNonRigid:
        np.save("atlas_prediction_non_rigid.npy", atlas_predictions)
        np.save("atlas_probabilitie_non_rigid.npy", atlas_probabilities)
    else:
        np.save("atlas_prediction_affine.npy", atlas_predictions)
        np.save("atlas_probabilitie_affine.npy", atlas_probabilities)
    return


def global_weighted_atlas(target, atlases):
    metricsT1w = metric.MeanSquaredError()
    metricsT2w = metric.MeanSquaredError()
    targetT1w = sitk.GetArrayFromImage(target.images[structure.BrainImageTypes.T1w])
    targetT2w = sitk.GetArrayFromImage(target.images[structure.BrainImageTypes.T2w])
    mask = sitk.GetArrayFromImage(target.images[structure.BrainImageTypes.BrainMask])
    metricsT1w.prediction = targetT1w[mask == 1]
    metricsT2w.prediction = targetT2w[mask == 1]

    mseT1w = []
    mseT2w = []

    # calculate similarity Mean SquaredError for each atlas
    for atlas in atlases:
        atlasT1w = sitk.GetArrayFromImage(atlas.images[structure.BrainImageTypes.T1w])
        atlasT2w = sitk.GetArrayFromImage(atlas.images[structure.BrainImageTypes.T2w])
        metricsT1w.reference = atlasT1w[mask == 1]
        metricsT2w.reference = atlasT2w[mask == 1]
        mseT1w.append(metricsT1w.calculate())
        mseT2w.append(metricsT2w.calculate())
    # calculate norm od MSE and averaging over T1w and T2w
    softmax_mse = special.softmax(-np.add(mseT1w, mseT2w))

    # Get the list of GroundTruth and converts the image in numpy format
    groundtruth_np = [sitk.GetArrayFromImage(atlas.images[structure.BrainImageTypes.GroundTruth]) for atlas in atlases]

    # Stack the images in a 4-D numpy array
    groundtruth_np = np.stack(groundtruth_np, axis=-1)

    # Compile the atlas by taking the most occurring label for each voxel
    atlas_np = weighted_mode(groundtruth_np, softmax_mse, axis=3)

    # write prediction and probabilities
    predictions = np.squeeze(atlas_np[0])
    probabilities = np.squeeze(atlas_np[1])

    return predictions, probabilities


def local_weighted_atlas(target, atlases):
    # Get the list of GroundTruth and converts the image in numpy format
    images_gt_np = [sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.GroundTruth]) for img in atlases]
    images_t1_np = [sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.T1w]) for img in atlases]
    images_t2_np = [sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.T2w]) for img in atlases]
    target_t1_np = sitk.GetArrayFromImage(target.images[structure.BrainImageTypes.T1w])
    target_t2_np = sitk.GetArrayFromImage(target.images[structure.BrainImageTypes.T2w])

    # Get local or global weights for each image
    seT1w = [np.square(np.subtract(target_t1_np, image_t1)) for image_t1 in images_t1_np]
    seT2w = [np.square(np.subtract(target_t2_np, image_t2)) for image_t2 in images_t2_np]
    weights_np = special.softmax(-np.add(seT1w, seT2w))

    # Get votes for each label, including individual weights
    atlas_np = weighted_mode(images_gt_np, weights_np)
    predictions = np.squeeze(atlas_np[0])
    probabilities = np.squeeze(atlas_np[1])

    return predictions, probabilities
