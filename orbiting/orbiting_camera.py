"""
threedb.controls.blender.orbiting_camera
===============================

Control the orbiting camera. 

"""

from typing import Any, Dict

import copy
from ...try_bpy import bpy, mathutils 
from ..base_control import PreProcessControl
from ...rendering.utils import lookat_viewport
from math import cos, sin, pi


class OrbitingCameraControl(PreProcessControl):
    """Control that changes how the camera orbits that will be used to render the image

    Continuous Dimensions:

    - ``object_x``: The x coordinate of the object. (range: ``[-1, 1]``)
    - ``object_y``: The y coordinate of the object. (range: ``[-1, 1]``)
    - ``object_z``: The z coordinate of the object. (range: ``[-1, 1]``)
    - ``radius``: The length of the radius of sphere orbit. (range: ``[3, 10]``)
    - ``phi``: The angle from positive z to the camera. (range: ``[-pi, pi]``)
    - ``theta``: The angle from positive x to the camera. (range: ``[-pi, pi]``)

    """
    def __init__(self, root_folder: str):
        continuous_dims = {
            'object_x': (-1, 1),
            'object_y': (-1, 1),
            'object_z': (-1, 1),
            'radius': (0.5, 10),
            'phi': (-pi, pi),
            'theta': (-pi, pi),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        args = copy.copy(control_args)

        camera = bpy.data.objects['Camera']
        obj = context['object']

        #move object
        obj.location = (args['object_x'], args['object_y'], args['object_z'])

        #change camera location
        camera.location = self.getCameraPosition(obj.location[0],
                                            obj.location[1],
                                            obj.location[2],
                                            args['radius'],
                                            args['phi'],
                                            args['theta'])
        
        #change camera direction
        #direction of camera
        direction = obj.location-camera.location
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')
        # assume we're using euler rotation
        camera.rotation_euler = rot_quat.to_euler()

    def unapply(self, context: Dict[str, Any]) -> None:
        pass
    
    def getCameraPosition(self, object_x, object_y, object_z, rho, phi, theta):
        camera_x = object_x + (rho * sin(phi) * cos(theta))
        camera_y = object_y + (rho * sin(phi) * sin(theta))
        camera_z = object_z + (rho * cos(phi))
        return camera_x, camera_y, camera_z

Control = OrbitingCameraControl
