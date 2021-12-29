#!/bin/bash


./upool_training.sh 5ch 1
./depth_training.sh 5ch 1
# cp -r ./experiment3_depth/GFCNC-dsGISLES2018-idPre5Ch ./experiment5_cmp/GFCNC-dsGISLES2018-idPre5Ch 
./cmp_training.sh 5ch 1


./upool_training.sh 5ch 2
./depth_training.sh 5ch 2
cp -r ./experiment3_depth/GFCNC-dsGISLES2018-idPre5Ch2 ./experiment5_cmp/GFCNC-dsGISLES2018-idPre5Ch2
./cmp_training.sh 5ch 2 2 


./upool_training.sh 5ch 3
./depth_training.sh 5ch 3
cp -r ./experiment3_depth/GFCNC-dsGISLES2018-idPre5Ch3 ./experiment5_cmp/GFCNC-dsGISLES2018-idPre5Ch3
./cmp_training.sh 5ch 3


./upool_training.sh 5ch 4
./depth_training.sh 5ch 4
cp -r ./experiment3_depth/GFCNC-dsGISLES2018-idPre5Ch4 ./experiment5_cmp/GFCNC-dsGISLES2018-idPre5Ch4 
./cmp_training.sh 5ch 4













