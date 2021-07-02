#!/bin/sh

cat experiments/changing_viewpoints/cup_mug/non_robust/results2/details.log >> experiments/changing_viewpoints/cup_mug/non_robust/results/details_non_rob.log

cp -a experiments/changing_viewpoints/cup_mug/non_robust/results2/images/. experiments/changing_viewpoints/cup_mug/non_robust/results/images

rm -r experiments/changing_viewpoints/cup_mug/non_robust/results2