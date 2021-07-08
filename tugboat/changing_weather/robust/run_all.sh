#!/bin/sh
for i in $(ls robust_models)
do
    #change model to imagenet_l2_3_0.pt
    sed -i 's@path_to_model:.*@path_to_model: "./robust_models/'$i'"@' ./tugboat/changing_weather/robust/base.yaml
    #run model
    eval "$(conda shell.bash hook)"
    conda activate threedb
    mkdir -p ./tugboat/changing_weather/robust/results

    #run master
    threedb_master data_tugboat ./tugboat/changing_weather/robust/changing_weather.yaml ./tugboat/changing_weather/robust/results 5555 &  PIDM=$!

    #run worker
    gnome-terminal -- bash -c "source ~/miniconda3/etc/profile.d/conda.sh; conda activate threedb; threedb_workers 2 data_tugboat 5555;" &  PIDW=$!

    #concurrently
    wait $PIDM
    wait $PIDW

    #rename details.log file
    mv ./tugboat/changing_weather/robust/results/details.log ./tugboat/changing_weather/robust/results/details_$i.log
done







