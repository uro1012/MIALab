"""The data structure module holds model classes."""
import enum

import pymia.data.conversion as conversion
import SimpleITK as sitk


class BrainImageTypes(enum.Enum):
    """Represents human readable image types."""
    T1w = 1  #: The T1-weighted image.
    T2w = 2  #: The T2-weighted image.
    GroundTruth = 3  #: The ground truth image.
    BrainMask = 4  #: The brain mask image.
    RegistrationTransform = 5  #: The affine registration transformation
    RegistrationParameterMap = 6  #: The non-rigid registration transformation


class BrainImage:
    """Represents a brain image."""
    def __init__(self, id_: str, path: str, images: dict, transformation: sitk.Transform, parameterMap: sitk.ParameterMap):
        """Initializes a new instance of the BrainImage class.

        Args:
            id_ (str): An identifier.
            path (str): Full path to the image directory.
            images (dict): The images, where the key is a :py:class:`BrainImageTypes` and the value is a SimpleITK image.
        """
        self.id_ = id_
        self.path = path
        self.images = images
        self.transformation = transformation
        self.parameterMap = parameterMap

        # ensure we have an image to get the image properties
        if len(images) == 0:
            raise ValueError('No images provided')

        self.image_properties = conversion.ImageProperties(self.images[list(self.images.keys())[0]])
        self.feature_images = {}
        self.feature_matrix = None  # a tuple (features, labels),
        # where the shape of features is (n, number_of_features) and the shape of labels is (n, 1)
        # with n being the amount of voxels
