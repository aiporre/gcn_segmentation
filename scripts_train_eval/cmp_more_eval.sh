#!/bin/bash

channels=$1
fold=$2
tfold=$fold
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
#  rm data/gisles/processed/TRAINING/gendo_*.pt
  tid="4Ch$fold"
  echo "PointNet with soft Dice loss"
  # python training.py -s GISLES2018 -n PointNet -b 4 -c FLsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 200 -f $tfold --mod TMAX CBF CBV MTT --postnorm False -X True -D experiment5_cmp >> "experiment5_cmp/gfcnc_${tid}_eval.log"
elif [[ $channels == "5ch" ]]
then
  echo " Deleting files gendo_"
  # rm data/gisles2018/processed/TRAINING/gendo_*.pt
  tid="5Ch$fold"
  echo "PointNet training"
  python training.py -s GISLES2018 -n PointNet -b 4 -c FLsigmoid --useful True -t True -N 10 --id $tid -lr 1E-7  -g 300 -f $tfold -X True -D experiment5_cmp >> "experiment5_cmp/pointnet_${tid}_eval.log"
  echo "Unet training: "
  python training.py -s ISLES2018 -n UNet -b 4 -c FL --useful True -t True -N 10 --id $tid -lr 1E-7  -g 300 -f $tfold -X True -D experiment5_cmp >> "experiment5_cmp/unet_${tid}_eval.log"
  echo "FCN training: "
  python training.py -s ISLES2018 -n FCN -b 4 -c FLsigmoid --useful True -t True -N 10 --id $tid -lr 1E-7  -g 300 -f $tfold -X True -D experiment5_cmp >> "experiment5_cmp/fcn_${tid}_eval.log"

elif [[ $channels == "1chCBV" ]]
then
  echo " Deleting files gendo_"
  echo " running with 1 channel in CBV"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  tid="1ChCBV$fold"
  echo "GFCNC with soft Dice loss"
  # python training.py -s GISLES2018 -n GFCNC -b 4 -c DLsigmoid -t True -N 10 --id $tid -lr 1E-5  -g 100 -f $tfold --mod CBV --postnorm False -X True -D experiment5_cmp >> "experiment5_cmp/gfcnc_${tid}_eval.log"
elif [[ $channels == "1chCBF" ]]
then
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo " running with 1 channel in CBF"
  echo " training losses case GFCNC"
  echo "GFCNC with soft Dice loss"
  tid="1ChCBF$fold"
  echo "GFCNC with soft Dice loss"
  # python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod CBF --postnorm False -X True -D experiment5_cmp >> "experiment5_cmp/gfcnc_${tid}_eval.log"
else
  echo  " nothing to run exit"
  exit 1
fi


