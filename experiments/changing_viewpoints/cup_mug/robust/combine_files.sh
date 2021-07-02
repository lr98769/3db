#!/bin/sh

cat experiments/changing_viewpoints/cup_mug/robust/results2/details_imagenet_l2_3_0.pt.log >> experiments/changing_viewpoints/cup_mug/robust/results/details_imagenet_l2_3_0.pt.log

cat experiments/changing_viewpoints/cup_mug/robust/results2/details_imagenet_linf_4.pt.log >> experiments/changing_viewpoints/cup_mug/robust/results/details_imagenet_linf_4.pt.log

cat experiments/changing_viewpoints/cup_mug/robust/results2/details_imagenet_linf_8.pt.log >> experiments/changing_viewpoints/cup_mug/robust/results/details_imagenet_linf_8.pt.log

rm -r experiments/changing_viewpoints/cup_mug/robust/results2