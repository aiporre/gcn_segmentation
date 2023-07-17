#!/bin/bash

fold=$1
tfold=$fold
if [[ $fold == "1" ]]
then
  echo "WARNING: fold 1 is set to empty string!"
  fold=""
fi

tid="Pre5Ch$fold"
python training.py -s GISLES2018 -n GFCNC -b 7 -c DCSsigmoid  -t True -N 15 --id $tid -lr 1E-6 -X True -g 200 -f $tfold --postnorm False -D models/isles2018 >> "models/isles2018/gfcnc_${tid}_graph.log"
