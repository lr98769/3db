"""
custom_controls.sun
===============================

Control the sun in the environment. 

"""

from typing import Any, Dict

import copy
from threedb.try_bpy import bpy, mathutils 
from threedb.controls.base_control import PreProcessControl
from threedb.rendering.utils import lookat_viewport
from math import pi


class SunControl(PreProcessControl):
    """Control that changes how the sun is positioned and how the sun rays appear

    Continuous Dimensions:

    - ``size``: Angular diameter of the sun disc (in radians). (range: ``[0, pi]``)
    - ``intensity``: Multiplier for sun disc lighting. (range: ``[0, 1]``)
    - ``elevation``: Rotation of the sun from the horizon (in radians). (range: ``[0, pi]``)
    - ``rotation``: Rotation of the sun around the zenith (in radians). (range: ``[0, 2pi]``)
    - ``altitude``: The distance from sea level to the location of the camera. For example, if the camera is placed on a beach then a value of 0 should be used. However, if the camera is in the cockpit of a flying airplane then a value of 10 km will be more suitable. Note, this is limited to 60 km because the mathematical model only accounts for the first two layers of the earth’s atmosphere (which ends around 60 km). (range: ``[0, 60]``)
    - ``air``: Density of air molecules (0: no air, 1: clear day atmosphere, 2:highly polluted day). (range: ``[0, 10]``)
    - ``dust``: Density of dust and water droplets. (0: no dust, 1:clear day atmosphere, 5:city like atmosphere, 10:hazy day) (range: ``[0, 10]``)
    - ``ozone``: Density of ozone molecules; useful to make the sky appear bluer (0: no ozone, 1: clear day atmosphere, 2:city like atmosphere). (range: ``[0, 10]``)
    - ``background_strength``: Strength of light emitted by background. (range: ``[0, 1]``)

    Refer to https://docs.blender.org/manual/en/latest/render/shader_nodes/textures/sky.html for more details.

    """
    def __init__(self, root_folder: str):
        continuous_dims = {
            'size': (0, pi),
            'intensity': (0, 1),
            'elevation': (0, pi),
            'rotation': (0, 2*pi),
            'altitude': (0, 60),
            'air': (0, 10),
            'dust': (0, 10),
            'ozone': (0, 10),
            'background_strength': (0, 1),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        args = copy.copy(control_args)

        elevation, rotation = self.changeElevationRotation(args['elevation'], args['rotation'])

        #set world to use nodes
        bpy.context.scene.world.use_nodes = True

        #get world nodes
        world_nodes = bpy.context.scene.world.node_tree.nodes

        if (world_nodes.get("Sky Texture") == None):
            #make sky texture nodes
            sky_node = world_nodes.new(type="ShaderNodeTexSky")
            background_node = world_nodes["Background"]
            #add link background node to sky texture node
            bpy.data.worlds['World'].node_tree.links.new(background_node.inputs[0], sky_node.outputs[0])

        sky_node = world_nodes.get("Sky Texture")

        #set attributes of sky texture node
        sky_node.sun_size = args['size']
        sky_node.sun_intensity = args['intensity']
        sky_node.sun_elevation = elevation
        sky_node.sun_rotation = rotation
        sky_node.altitude = args['altitude']
        sky_node.air_density = args['air']
        sky_node.dust_density = args['dust']
        sky_node.ozone_density = args['ozone']

        #get background node
        background_node = world_nodes["Background"]

        #change brightness of the sky
        background_node.inputs[1].default_value = args['background_strength']

    def unapply(self, context: Dict[str, Any]) -> None:
        pass
    
    #to allow for 180 degree sun, since elevation has a maximum of 90
    def changeElevationRotation(self, elevation, rotation):
        if (elevation > pi/2):
            elevation = pi - elevation
            rotation += pi
        return elevation, rotation

Control = SunControl
