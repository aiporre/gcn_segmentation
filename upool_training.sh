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
  echo " Deleting files gendo_"
  # rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo "Training upool proportional vs topk"
  echo "Running upool proportional"
  tid="uPre4Ch$fold"
  python training.py -s GISLES2018 -n GFCNE -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-7 -g 100 --mod TMAX CBF CBV MTT --postnorm False -D experiment4_upool -W True>> "experiment4_upool/gfcne_$tid.log"
  echo "Running upool topk"
  tid="kPre4Ch$fold"
  python training.py -s GISLES2018 -n GFCNG -b 8 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-4 -g 100 --mod TMAX CBF CBV MTT --postnorm False -D experiment4_upool >> "experiment4_upool/gfcng_$tid.log"
  echo "Running no unpool and pooling"
  tid="nPre4Ch$fold"
  python training.py -s GISLES2018 -n GFCNF -b 10 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod TMAX CBF CBV MTT --postnorm False -D experiment4_upool >> "experiment4_upool/gfcnf_$tid.log"
elif [[ $channels == "5ch" ]]
then
  echo "Running 5 channels"
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo "Training upool proportional vs topk"
  echo "Running upool proportional"
  tid="uPre5Ch$fold"
  python training.py -s GISLES2018 -n GFCNE -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --postnorm False -D experiment4_upool -W True>> "experiment4_upool/gfcnc_$tid.log"
  echo "Running upool topk"
  tid="kPre5Ch$fold"
  python training.py -s GISLES2018 -n GFCNG -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --postnorm False -D experiment4_upool >> "experiment4_upool/gfcnd_$tid.log"
elif [[ $channels == "1chCBV" ]]
then
  echo " running with 1 channel in CBV"
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo "Training upool proportional vs topk"
  tid="uPre1ChCBV$fold"
  echo "Running upool"
  python training.py -s GISLES2018 -n GFCNE -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod CBV --postnorm False -D experiment4_upool -W True>> "experiment4_upool/gfcnc_$tid.log"
  echo "Running topk"
  tid="kPre1ChCBV$fold"
  python training.py -s GISLES2018 -n GFCNG -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod CBV --postnorm False -D experiment4_upool >> "experiment4_upool/gfcnd_$tid.log"
#
elif [[ $channels == "1chCBF" ]]
then
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo " running with 1 channel in CBF"
  echo "Training upool proportional vs topk"
  echo "Running upool"
  tid="uPre1ChCBF$fold"
  python training.py -s GISLES2018 -n GFCNE -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod CBF --postnorm False -D experiment4_upool -W True>> "experiment4_upool/gfcnc_$tid.log"
  echo "Running topk"
  tid="kPre1ChCBF$fold"
  python training.py -s GISLES2018 -n GFCNG -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6 -g 100 --mod CBF --postnorm False -D experiment4_upool >> "experiment4_upool/gfcnd_$tid.log"
else
  echo  " nothing to run exit"
  exit 1
fi


