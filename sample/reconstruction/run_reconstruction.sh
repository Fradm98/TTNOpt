#!/bin/bash

mkdir sample/normal_distribution_data
mkdir sample/normal_distribution_data/cov
mkdir sample/normal_distribution_data/before
mkdir sample/normal_distribution_data/after

python sample/reconstruction/create_cov.py

python sample/reconstruction/create_MPS.py

ft sample/reconstruction/normal_distribution_init.yml
ft sample/reconstruction/normal_distribution.yml

python sample/reconstruction/visualize_edges.py