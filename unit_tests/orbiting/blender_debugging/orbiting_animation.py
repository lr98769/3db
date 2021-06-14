import bpy
from math import cos, sin, pi


bpy.ops.mesh.primitive_cube_add()
bpy.ops.object.camera_add()

camera = bpy.data.objects["Camera"]
cube = bpy.data.objects["Cube"]


def getCameraPosition(object_x, object_y, object_z, rho, phi, theta):
    camera_x = object_x + (rho * sin(phi) * cos(theta))
    camera_y = object_y + (rho * sin(phi) * sin(theta))
    camera_z = object_z + (rho * cos(phi))
    return camera_x, camera_y, camera_z

def rotateCamera(rho,phi, theta):
    #change camera location
    camera.location = getCameraPosition(cube.location[0],
                                        cube.location[1],
                                        cube.location[2],
                                        rho,
                                        phi,
                                        theta)
    #change camera direction
    #direction of camera
    direction = cube.location-camera.location
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    camera.rotation_euler = rot_quat.to_euler()

phi_list = [pi/4, pi/2, pi*0.75]
theta_list = [0, pi/2, pi, pi*1.5, pi*2]

i=0
for phi in phi_list:
    for theta in theta_list:
        rotateCamera(10, phi, theta)
        camera.keyframe_insert(data_path="location", frame=i*5)
        camera.keyframe_insert(data_path="rotation_euler", frame=i*5)
        i+=1
                            




