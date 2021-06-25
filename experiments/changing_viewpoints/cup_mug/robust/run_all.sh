#!/bin/sh
for i in $(ls robust_models)
do
    #change model to imagenet_l2_3_0.pt
    sed -i 's@path_to_model:.*@path_to_model: "./robust_models/'$i'"@' changing_viewpoints/cup_mug/robust/base.yaml

    #run model
    eval "$(conda shell.bash hook)"
    conda activate threedb
    mkdir -p ./changing_viewpoints/cup_mug/robust/results

    #run master
    threedb_master data_cup ./changing_viewpoints/cup_mug/robust/changing_viewpoints.yaml ./changing_viewpoints/cup_mug/robust/results 5555 &  PIDM=$!


    #run worker
    gnome-terminal -- bash -c "source ~/miniconda3/etc/profile.d/conda.sh; conda activate threedb; threedb_workers 1 data_cup 5555;" &  PIDW=$!

    #concurrently
    wait $PIDM
    wait $PIDW

    #rename details.log file
    mv ./changing_viewpoints/cup_mug/robust/results/details.log ./changing_viewpoints/cup_mug/robust/results/details_$i.log
done







