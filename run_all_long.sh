#!/bin/bash


./cmp_training.sh 5ch 3


./upool_training.sh 5ch 4
./depth_training.sh 5ch 4
cp -r ./experiment3_depth/GFCNC-dsGISLES2018-idPre5Ch4 ./experiment5_cmp/GFCNC-dsGISLES2018-idPre5Ch4 
./cmp_training.sh 5ch 4

./depth_pos_training.sh 5ch 1 
./depth_pos_training.sh 5ch 2
./depth_pos_training.sh 5ch 3
./depth_pos_training.sh 5ch 4

./gfcne_training.sh 5ch 1
./gfcne_training.sh 5ch 2
./gfcne_training.sh 5ch 3

./cmp_more_training.sh 5ch 1
./cmp_more_training.sh 5ch 2
./cmp_more_training.sh 5ch 3
./cmp_more_training.sh 5ch 4
