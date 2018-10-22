#!/bin/bash

#prepare training vfh features
TRAINDATA=training_data/models/
./prepare_data $TRAINDATA

#build search tree
TRAINFEATURE=training_data/features/
./build_tree $TRAINFEATURE

#prepare test vfh features
TESTDATA=test_data/cup_test_sample_2.pcd
./scene_test $TESTDATA
