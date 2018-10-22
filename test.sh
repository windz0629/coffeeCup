#!/bin/bash

# Test.pcd file
# DATA=training_data/features/cloud_view_5_vfh.pcd
DATA=test_data/cluster_0_vfh.pcd

# Inlier distance threshold
thresh=60

# Get the closest K nearest neighbors
k=12

./nearest_neighbors -k $k -thresh $thresh $DATA
