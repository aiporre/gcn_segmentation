import numpy as np
import numpy_groupies as npg

from .form_feature_extraction import FormFeatureExtraction


def extract_features(segmentation, image, form_features=None):
    features = FormFeatureExtraction(segmentation).get_features(form_features)

    group_idx = segmentation.flatten()

    # Prepend mean color to form features.
    if image.shape[2] == 1:
        mean = npg.aggregate(group_idx, image.flatten(), func='mean')
        mean = np.reshape(mean, (-1, 1))
        features = np.concatenate((mean, features), axis=1)
    elif image.shape[2] == 3:
        r = npg.aggregate(group_idx, image[:, :, 0:1].flatten(), func='mean')
        r = np.reshape(r, (-1, 1))
        g = npg.aggregate(group_idx, image[:, :, 1:2].flatten(), func='mean')
        g = np.reshape(g, (-1, 1))
        b = npg.aggregate(group_idx, image[:, :, 2:3].flatten(), func='mean')
        b = np.reshape(b, (-1, 1))
        features = np.concatenate((r, g, b, features), axis=1)
    else:
        raise ValueError

    return features.astype(np.float32)


def extract_features_fixed(form_features=None):
    def _extract(segmentation, image):
        return extract_features(segmentation, image, form_features)

    return _extract
