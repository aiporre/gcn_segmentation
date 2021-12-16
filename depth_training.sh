#!/bin/bash

channels=$1
fold=$2
if [[ $fold == "1" ]]
then
  echo "WARNING: fold 1 is set to empty string!"
  fold=""
fi

conda activate gfcn
if [[ $channels == "4ch" ]]
then
  echo "running 4ch"
  echo "Deleting files gendo_"
#  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  tid="Pre4Ch$fold"
  echo "GFCNC with soft Dice loss"
#  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod TMAX CBF CBV MTT --postnorm False -D experiment3_depth >> "experiment3_depth/gfcnc_$tid.log"
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod TMAX CBF CBV MTT --postnorm False -D experiment3_depth >> "experiment3_depth/gfcnb_$tid.log"
  echo "GFCNA with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod TMAX CBF CBV MTT --postnorm False -D experiment3_depth >> "experiment3_depth/gfcna_$tid.log"

elif [[ $channels == "5ch" ]]
then
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  tid="Pre5Ch$fold"
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --postnorm False -D experiment3_depth >> "experiment3_depth/gfcnc_$tid.log"
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --postnorm False -D experiment3_depth >> "experiment3_depth/gfcnb_$tid.log"
  echo "GFCNA with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --postnorm False -D experiment3_depth >> "experiment3_depth/gfcna_$tid.log"
elif [[ $channels == "1chCBV" ]]
then
  echo " Deleting files gendo_"
  echo " running with 1 channel in CBV"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  tid="Pre1ChCBV$fold"
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod CBV --postnorm False -D experiment3_depth >> "experiment3_depth/gfcnc_$tid.log"
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod CBV --postnorm False -D experiment3_depth >> "experiment3_depth/gfcnb_$tid.log"
  echo "GFCNA with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod CBV --postnorm False -D experiment3_depth >> "experiment3_depth/gfcna_$tid.log"
#
elif [[ $channels == "1chCBF" ]]
then
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo " running with 1 channel in CBF"
  echo " training losses case GFCNC"
  echo "GFCNC with soft Dice loss"
  tid="Pre1ChCBF$fold"
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod CBF --postnorm False -D experiment3_depth >> "experiment3_depth/gfcnc_$tid.log"
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod CBF --postnorm False -D experiment3_depth >> "experiment3_depth/gfcnb_$tid.log"
  echo "GFCNA with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod CBF --postnorm False -D experiment3_depth >> "experiment3_depth/gfcna_$tid.log"
else
  echo  " nothing to run exit"
  exit 1
fi


