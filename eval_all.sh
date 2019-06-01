#!/bin/bash


./upool_eval.sh 5ch 1
./upool_eval.sh 5ch 2
./upool_eval.sh 5ch 3
./upool_eval.sh 5ch 4


./depth_pos_eval.sh 5ch 1
./depth_pos_eval.sh 5ch 2
./depth_pos_eval.sh 5ch 3
./depth_pos_eval.sh 5ch 4

./depth_eval.sh 5ch 1
./depth_eval.sh 5ch 2
./depth_eval.sh 5ch 3
./depth_eval.sh 5ch 4

./cmp_eval.sh 5ch 1
./cmp_eval.sh 5ch 2
./cmp_eval.sh 5ch 3
./cmp_eval.sh 5ch 4

./gfcne_eval.sh 5ch 1
./gfcne_eval.sh 5ch 2
./gfcne_eval.sh 5ch 3
./gfcne_eval.sh 5ch 4

./cmp_more_eval.sh 5ch 1
./cmp_more_eval.sh 5ch 2
./cmp_more_eval.sh 5ch 3
./cmp_more_eval.sh 5ch 4
