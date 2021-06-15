"""
custom_controls.rain
====================================

Apply rain to renderings. 

"""
import copy
from typing import Any, Dict
import numpy as np
import torch as ch
import imgaug.augmenters as iaa
from threedb.controls.base_control import PostProcessControl

class RainControl(PostProcessControl):
    """
    Applies the rain effects from imgaug. 

    Discrete Dimensions:

    - ``n_layers_of_rain``: Number of layers of rain. (range: ``{1, 2, 3}``)

    Continuous Dimensions:

    - ``speed``: Length of rain streaks. (range: ``[0.1, 0.9]``) (Recommended: ``[0.3, 0.9]``)
    - ``drop_size``: Width of rain streaks. (range: ``[0.1, 0.9]``) (Recommended: ``[0.1, 0.3]``)

    """
    def __init__(self, root_folder: str):
        discrete_dims = {
            'n_layers_of_rain': [1, 2, 3],
        }
        continuous_dims = {
            'speed': (0, 1),
            'drop_size': (0.1, 1),
        }
        super().__init__(root_folder,
                         discrete_dims=discrete_dims, 
                         continuous_dims=continuous_dims)

    def apply(self, render: ch.Tensor, control_args: Dict[str, Any]) -> ch.Tensor:
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        args = copy.copy(control_args)
        img = render.numpy()
        img = img.transpose(1, 2, 0)
        img = (img * 255).astype('uint8')

        aug = iaa.Rain( nb_iterations = args['n_layers_of_rain'], 
                        speed = args['speed'], 
                        drop_size = args['drop_size'])

        augmented = aug.augment_image(img[:,:,:3])
        augmented = np.concatenate((augmented, img[:, :, 3:4]), axis=2)
        augmented = augmented.transpose(2, 0, 1)
        augmented = augmented.astype('float32') / 255

        return ch.from_numpy(augmented)

    def unapply(self, context: Dict[str, Any]) -> None:
        pass

Control = RainControl
