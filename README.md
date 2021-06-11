# 3db

## 1. Orbiting 
Orbits the camera around the object. User only has to set the radius, phi and theta. 

Varying phi             |  Varying theta
:-------------------------:|:-------------------------:
![alt text](https://github.com/lr98769/3db/blob/dev/orbiting/unit_test/dashboard_phi.gif)  |  ![alt text](https://github.com/lr98769/3db/blob/dev/orbiting/unit_test/dashboard_theta.gif)

**To use the orbiting module:**
1. Include orbiting_camera.py in /envs/threedb/lib/python3.7/site-packages/threedb/controls/blender/
2. Include threedb.controls.blender.orbiting_camera in the yaml config file

**To run unit test for orbiting module:**
1. Set up threedb with the orbiting module with the above 2 steps
2. Download Primer data from (https://www.dropbox.com/s/2gdprhp8jvku4zf/threedb_starting_kit.tar.gz?dl=0) and place "studioX_Stage.blend" in data/blender_environments (images generated may be different due to rotation and transformation of the background model)
2. Run these two commands concurrently in different terminals (run threedb_master first)

*Run in first terminal*
```
conda activate threedb
cd orbiting/unit_test
threedb_master ../../data orbit_around.yaml results 5555
```
*Run in second terminal*
```
conda activate threedb
cd orbiting/unit_test
threedb_workers 1 ../../data 5555
```
*To analyse:*
```
conda activate threedb
cd orbiting/unit_test
python -m threedboard results
```

## 2. Rain
Post-process rain onto render images. 

Varying speed             |  Varying drop size   |  Varying layers of rain
:-------------------------:|:-------------------------:|:-------------------------:
![alt text](https://github.com/lr98769/3db/blob/dev/rain/unit_test/dashboard_speed.gif)  |  ![alt text](https://github.com/lr98769/3db/blob/dev/rain/unit_test/dashboard_drop_size.gif)   |   ![alt text](https://github.com/lr98769/3db/blob/dev/rain/unit_test/dashboard_layers.gif)

**To use the rain module:**
1. Install the imgaug package with the following lines of code:
```
pip install git+https://github.com/aleju/imgaug.git
pip install imagecorruptions
```
2. Include rain.py in /envs/threedb/lib/python3.7/site-packages/threedb/controls/blender/
3. Include threedb.controls.blender.rain in the yaml config file

**To run unit test for rain module:**
1. Set up threedb with the rain module with the above 3 steps
2. Download Primer data from (https://www.dropbox.com/s/2gdprhp8jvku4zf/threedb_starting_kit.tar.gz?dl=0) and place "studioX_Stage.blend" in data/blender_environments (images generated may be different due to rotation and transformation of the background model)
3. Run these two commands concurrently in different terminals (run threedb_master first)

*Run in first terminal*
```
conda activate threedb
cd rain/unit_test
threedb_master ../../data rain.yaml results 5555
```
*Run in second terminal*
```
conda activate threedb
cd rain/unit_test
threedb_workers 1 ../../data 5555
```
*To analyse:*
```
conda activate threedb
cd rain/unit_test
python -m threedboard results
```