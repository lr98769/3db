# 3db
Custom control modules for 3db. (Before using any of the custom controls below, make sure 3db is installed.)

Link to 3db repository: https://github.com/3db/3db

## 1. Orbiting 
Pre-processing control module that orbits the camera around the object. User only has to set the radius, phi and theta. 

Varying phi             |  Varying theta
:-------------------------:|:-------------------------:
![alt text](https://github.com/lr98769/3db/blob/dev/unit_tests/orbiting/gifs/dashboard_phi.gif)  |  ![alt text](https://github.com/lr98769/3db/blob/dev/unit_tests/orbiting/gifs/dashboard_theta.gif)

**To use the orbiting module:**
1. Include custom_controls.orbiting_camera in the yaml config file

**To run unit test for orbiting module:**
1. Download Primer data from (https://www.dropbox.com/s/2gdprhp8jvku4zf/threedb_starting_kit.tar.gz?dl=0) and place "studioX_Stage.blend" in data/blender_environments (images generated may be different due to rotation and transformation of the background model)
2. Run these two commands concurrently in different terminals (run threedb_master first)

*Run in first terminal*
```
conda activate threedb
threedb_master data unit_tests/orbiting/orbit_around.yaml unit_tests/orbiting/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard unit_tests/orbiting/results
```

## 2. Rain
Post-processing control module that adds rain to rendered images. 

Varying speed             |  Varying drop size   |  Varying layers of rain
:-------------------------:|:-------------------------:|:-------------------------:
![alt text](https://github.com/lr98769/3db/blob/dev/unit_tests/rain/gifs/dashboard_speed.gif)  |  ![alt text](https://github.com/lr98769/3db/blob/dev/unit_tests/rain/gifs/dashboard_drop_size.gif)   |   ![alt text](https://github.com/lr98769/3db/blob/dev/unit_tests/rain/gifs/dashboard_layers.gif)

**To use the rain module:**
1. Install the imgaug package with the following lines of code:
```
pip install git+https://github.com/aleju/imgaug.git
pip install imagecorruptions
```
2. Include custom_controls.rain in the yaml config file

**To run unit test for rain module:**
1. Install the imgaug package with the following lines of code:
```
pip install git+https://github.com/aleju/imgaug.git
pip install imagecorruptions
```
2. Download Primer data from (https://www.dropbox.com/s/2gdprhp8jvku4zf/threedb_starting_kit.tar.gz?dl=0) and place "studioX_Stage.blend" in data/blender_environments (images generated may be different due to rotation and transformation of the background model)
3. Run these two commands concurrently in different terminals (run threedb_master first)

*Run in first terminal*
```
conda activate threedb
threedb_master data unit_tests/rain/rain.yaml unit_tests/rain/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard unit_tests/rain/results
```

## 3. Sun
Pre-processing control module that adds a sun into the environment and modifies the sun's properties.  

Varying elevation             |  Varying rotation
:-------------------------:|:-------------------------:
![alt text](https://github.com/lr98769/3db/blob/dev/unit_tests/sun/gifs/dashboard_elevation.gif)  |  ![alt text](https://github.com/lr98769/3db/blob/dev/unit_tests/sun/gifs/dashboard_rotation.gif)

**To use the sun module:**
1. Include custom_controls.sun in the yaml config file

**To run unit test for rain module:**
1. Download Primer data from (https://www.dropbox.com/s/2gdprhp8jvku4zf/threedb_starting_kit.tar.gz?dl=0) and place "studioX_Stage.blend" in data/blender_environments (images generated may be different due to rotation and transformation of the background model)
2. Run these two commands concurrently in different terminals (run threedb_master first)

*Run in first terminal*
```
conda activate threedb
threedb_master data unit_tests/sun/sun.yaml unit_tests/sun/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard unit_tests/sun/results
```

## 4. Haze
Pre-processing control module that adds a haze into the environment.  

Varying Factor             |
:-------------------------:|
![alt text](https://github.com/lr98769/3db/blob/dev/unit_tests/haze/gifs/dashboard_fac.gif)  |  

**To use the sun module:**
1. Include custom_controls.haze in the yaml config file

**To run unit test for rain module:**
1. Download Primer data from (https://www.dropbox.com/s/2gdprhp8jvku4zf/threedb_starting_kit.tar.gz?dl=0) and place "studioX_Stage.blend" in data/blender_environments (images generated may be different due to rotation and transformation of the background model)
2. Run these two commands concurrently in different terminals (run threedb_master first)

*Run in first terminal*
```
conda activate threedb
threedb_master data unit_tests/haze/haze.yaml unit_tests/haze/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard unit_tests/haze/results
```

## Combined
Example of how all 4 modules can be used together.
Analysis folder contains a jupter notebook with useful functions for model analysis.

*Run in first terminal*
```
conda activate threedb
threedb_master data combined/combined.yaml combined/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard combined/results
```

## Testing Custom Models
Example of how trained models can be easily loaded and evaluated with 3db. 

*Run in first terminal*
```
conda activate threedb
threedb_master data unit_tests/load_model/load_model.yaml unit_tests/load_model/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard unit_tests/load_model/results
```

## Changing Viewpoints
Investigating if pixel perturbation robust models perform better on images of cups with varying viewpoints.
All robust models were downloaded from https://github.com/MadryLab/robustness

Setup:
1. Install robustness python module
2. Download imagent_l2_3_0.pt to ./changing_viewpoints/robust_models from https://github.com/MadryLab/robustness

# For the Mug

**To establish the baseline performance of a non-robust model:**
Evaluate the performance of a non-robust resnet50 model from torchvision

*Run in first terminal*
```
conda activate threedb
threedb_master data ./changing_viewpoints/mug/non_robust/changing_viewpoints.yaml ./changing_viewpoints/mug/non_robust/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard ./changing_viewpoints/non_robust/results 
```

**Evaluate performance of robust models:**
Evaluate the performance of a robust resnet50 model

Change the address to model in base.yaml to the address of the desired model to evaluate
'''path_to_model: './changing_viewpoints/robust_models/change_model_name''''

*Run in first terminal*
```
conda activate threedb
threedb_master data ./changing_viewpoints/mug/robust/changing_viewpoints.yaml ./changing_viewpoints/mug/robust/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard ./changing_viewpoints/mug/non_robust/results 
```

# For the Cup

**To establish the baseline performance of a non-robust model:**
Evaluate the performance of a non-robust resnet50 model from torchvision

*Run in first terminal*
```
conda activate threedb
threedb_master data_cup ./changing_viewpoints/cup/non_robust/changing_viewpoints.yaml ./changing_viewpoints/cup/non_robust/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data_cup 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard ./changing_viewpoints/cup/non_robust/results 
```

**Evaluate performance of robust models:**
Evaluate the performance of a robust resnet50 model

Change the address to model in base.yaml to the address of the desired model to evaluate
'''path_to_model: './changing_viewpoints/robust_models/change_model_name''''

*Run in first terminal*
```
conda activate threedb
threedb_master data_cup ./changing_viewpoints/cup/robust/changing_viewpoints.yaml ./changing_viewpoints/cup/robust/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data_cup 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard ./changing_viewpoints/cup/robust/results 
```

## Changing Viewpoints
Investigating if pixel perturbation robust models perform better on images of cups with varying viewpoints.
All robust models were downloaded from https://github.com/MadryLab/robustness

Setup:
1. Install robustness python module
2. Download imagent_l2_3_0.pt to ./changing_viewpoints/robust_models from https://github.com/MadryLab/robustness

**To establish the baseline performance of a non-robust model:**
Evaluate the performance of a non-robust resnet50 model from torchvision

*Run in first terminal*
```
conda activate threedb
threedb_master data ./changing_viewpoints/non_robust/changing_viewpoints.yaml ./changing_viewpoints/non_robust/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard ./changing_viewpoints/non_robust/results 
```

**Evaluate performance of robust models:**
Evaluate the performance of a robust resnet50 model

Change the address to model in base.yaml to the address of the desired model to evaluate
'''path_to_model: './changing_viewpoints/robust_models/change_model_name''''

*Run in first terminal*
```
conda activate threedb
threedb_master data ./changing_viewpoints/robust/changing_viewpoints.yaml ./changing_viewpoints/robust/results 5555
```
*Run in second terminal*
```
conda activate threedb
threedb_workers 1 data 5555
```
*To analyse:*
```
conda activate threedb
python -m threedboard ./changing_viewpoints/non_robust/results 
```

## Citation
```
@inproceedings{leclerc2021three,
    title={3DB: A Framework for Debugging Computer Vision Models},
    author={Guillaume Leclerc, Hadi Salman, Andrew Ilyas, Sai Vemprala, Logan Engstrom, Vibhav Vineet, Kai Xiao, Pengchuan Zhang, Shibani Santurkar, Greg Yang, Ashish Kapoor, Aleksander Madry},
    year={2021},
    booktitle={Arxiv preprint arXiv:2106.03805}
}
```