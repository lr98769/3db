#!/bin/sh
for i in $(ls robust_models)
do
    #change model to imagenet_l2_3_0.pt
    sed -i 's@path_to_model:.*@path_to_model: "./robust_models/'$i'"@' ./experiments/changing_viewpoints/ten_objects/robust/base.yaml

    #run model
    eval "$(conda shell.bash hook)"
    conda activate threedb
    mkdir -p ./experiments/changing_viewpoints/ten_objects/robust/results2

    #run master
    threedb_master data_new ./experiments/changing_viewpoints/ten_objects/robust/changing_viewpoints.yaml ./experiments/changing_viewpoints/ten_objects/robust/results2 5555 &  PIDM=$!


    #run worker
    gnome-terminal -- bash -c "source ~/miniconda3/etc/profile.d/conda.sh; conda activate threedb; threedb_workers 1 data_new 5555;" &  PIDW=$!

    #concurrently
    wait $PIDM
    wait $PIDW

    #rename details.log file
    mv ./experiments/changing_viewpoints/ten_objects/robust/results2/details.log ./experiments/changing_viewpoints/ten_objects/robust/results2/details_$i.log
done







