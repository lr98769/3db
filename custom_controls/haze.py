"""
custom_controls.haze
===============================

Control the haze in the environment. 

"""

from typing import Any, Dict

import copy
from threedb.try_bpy import bpy, mathutils 
from threedb.controls.base_control import PreProcessControl
from threedb.rendering.utils import lookat_viewport
from math import pi


class HazeControl(PreProcessControl):
    """Control that changes how the haze appears.

    Discrete Dimensions:

    - ``haze_falloff``: Curve function that controls the rate of change of the haze’s strength further and further into the distance. (range: ``{LINEAR, INVERSE, QUADRATIC}``) 

    Continuous Dimensions:

    - ``haze_start``: Distance from the camera at which the haze starts to fade in. (range: ``[0, 10]``) 
    - ``haze_depth``: Distance from start of the haze, that it fades in over. Objects further from the camera than Start + Depth are completely hidden by the haze. (range: ``[0, 10]``) 
    - ``haze_fac``:   Haze factor: How opaque the haze is (0 for transparent, 1 for opaque). (range: ``[0, 1]``) 

    Refer to https://docs.blender.org/manual/en/2.79/render/blender_render/world/mist.html for more details.

    """

    def __init__(self, root_folder: str):
        discrete_dims = {
            'haze_falloff': ["LINEAR", "INVERSE", "QUADRATIC"],
        }

        continuous_dims = {
            'haze_start': (0, 10),
            'haze_depth': (0, 10),
            'haze_fac': (0, 1),
        }
        
        super().__init__(root_folder, 
                        discrete_dims=discrete_dims, 
                        continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        args = copy.copy(control_args)

        #Render with cycles
        bpy.context.scene.render.engine = 'CYCLES'

        # 1. Enable the mist pass
        bpy.context.scene.view_layers["View Layer"].use_pass_mist = True

        # 2. Enable it in the camera properties under viewport settings
        bpy.data.objects['Camera'].data.show_mist = True

        # 3. Set the depth according to your scene
        bpy.context.scene.world.mist_settings.start = args["haze_start"]
        bpy.context.scene.world.mist_settings.depth = args["haze_depth"]
        bpy.context.scene.world.mist_settings.falloff = args["haze_falloff"]

        # 4. Render the image and use the mist pass with a mix node in the compositor
        bpy.context.scene.use_nodes = True
        scene_nodes = bpy.context.scene.node_tree.nodes
        scene_links = bpy.context.scene.node_tree.links
        
        if (bpy.context.scene.node_tree.nodes.get("Mix") == None):
            mix_node = scene_nodes.new(type="CompositorNodeMixRGB")
            render_layers_node = scene_nodes["Render Layers"]

            scene_links.new(mix_node.inputs[1], render_layers_node.outputs[0])
            scene_links.new(mix_node.inputs[2], render_layers_node.outputs[3])

            file_output_node = scene_nodes["exr_output"]
            scene_links.new(file_output_node.inputs[0], mix_node.outputs[0])

        mix_node = bpy.context.scene.node_tree.nodes.get("Mix")
        mix_node.inputs['Fac'].default_value = args["haze_fac"]

    def unapply(self, context: Dict[str, Any]) -> None:
        pass

Control = HazeControl
