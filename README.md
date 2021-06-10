# 3db

**To use the orbiting module:**
1. Include orbiting_camera.py in /envs/threedb/lib/python3.7/site-packages/threedb/controls/blender/ for module
2. Include threedb.controls.blender.orbiting_camera in the yaml config file

**To run unit test for orbiting module:**
1. Download Primer data from (https://www.dropbox.com/s/2gdprhp8jvku4zf/threedb_starting_kit.tar.gz?dl=0) and place "studioX_Stage.blend" in unit_test/blender_environments (images generated may be different due to rotation and transformation of the background model)
2. Run these two commands concurrently in different terminals (run threedb_master first)

*Run in first terminal*
```
conda activate threedb
cd unit_test
threedb_master data orbit_around.yaml results 5555
```
*Run in second terminal*
```
conda activate threedb
cd unit_test
threedb_workers 1 data 5555
```
*To analyse:*
```
conda activate threedb
cd unit_test
python -m threedboard results
```