#!/bin/sh

cat experiments/changing_weather/ten_objects/non_robust/results2/details.log >> experiments/changing_weather/ten_objects/non_robust/results/details_non_rob.log

cp -a experiments/changing_weather/ten_objects/non_robust/results2/images/. experiments/changing_weather/ten_objects/non_robust/results/images

rm -r experiments/changing_weather/ten_objects/non_robust/results2