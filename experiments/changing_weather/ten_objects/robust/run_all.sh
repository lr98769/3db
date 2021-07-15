#!/bin/sh
for i in $(ls robust_models)
do
    #change model to imagenet_l2_3_0.pt
    sed -i 's@path_to_model:.*@path_to_model: "./robust_models/'$i'"@' ./experiments/changing_weather/ten_objects/robust/base.yaml

    #run model
    eval "$(conda shell.bash hook)"
    conda activate threedb
    mkdir -p ./experiments/changing_weather/ten_objects/robust/results

    #run master
    threedb_master data_all ./experiments/changing_weather/ten_objects/robust/changing_weather.yaml ./experiments/changing_weather/ten_objects/robust/results 5555 &  PIDM=$!


    #run worker
    gnome-terminal -- bash -c "source ~/miniconda3/etc/profile.d/conda.sh; conda activate threedb; threedb_workers 1 data_all 5555;" &  PIDW=$!

    #concurrently
    wait $PIDM
    wait $PIDW

    #rename details.log file
    mv ./experiments/changing_weather/ten_objects/robust/results/details.log ./experiments/changing_weather/ten_objects/robust/results/details_$i.log
done







