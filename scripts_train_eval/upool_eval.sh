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
  echo " Deleting files gendo_"
  # rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo "Training upool proportional vs topk"
  echo "Running upool proportional"
  tid="uPre4Ch$fold"
  python training.py -s GISLES2018 -n GFCNE -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-7  -g 100 -f $tfold --mod TMAX CBF CBV MTT --postnorm False -X True -D experiment4_upool -W True>> "experiment4_upool/gfcne_${tid}_eval.log"
  echo "Running upool topk"
  tid="kPre4Ch$fold"
  python training.py -s GISLES2018 -n GFCNG -b 10 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-4  -g 100 -f $tfold --mod TMAX CBF CBV MTT --postnorm False -X True -D experiment4_upool >> "experiment4_upool/gfcng_${tid}_eval.log"
  echo "Running no unpool and pooling"
  tid="nPre4Ch$fold"
  python training.py -s GISLES2018 -n GFCNF -b 2 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod TMAX CBF CBV MTT --postnorm False -X True -D experiment4_upool >> "experiment4_upool/gfcnf_${tid}_eval.log"
elif [[ $channels == "5ch" ]]
then
  echo "Running 5 channels"
  echo " Deleting files gendo_"
  # rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo "Training upool proportional vs topk"
  echo "Running upool proportional"

  tid="uPre5Ch$fold"
  python training.py -s GISLES2018 -n GFCNE -b 4 -c DCSsigmoid -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --postnorm False -X True -D experiment4_upool -W True>> "experiment4_upool/gfcne_${tid}_eval.log"
  echo "Running upool topk"
  tid="kPre5Ch$fold"
  python training.py -s GISLES2018 -n GFCNG -b 8 -c DCSsigmoid -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --postnorm False -X True -D experiment4_upool >> "experiment4_upool/gfcng_${tid}_eval.log"
  echo "Running no unpool and pooling"
  tid="nPre5Ch$fold"
  python training.py -s GISLES2018 -n GFCNF -b 2 -c DCSsigmoid -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --postnorm False -X True -D experiment4_upool >> "experiment4_upool/gfcnf_${tid}_eval.log"
  
elif [[ $channels == "1chCBV" ]]
then
  echo " running with 1 channel in CBV"
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo "Training upool proportional vs topk"
  tid="uPre1ChCBV$fold"
  echo "Running upool"
  python training.py -s GISLES2018 -n GFCNE -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod CBV --postnorm False -X True -D experiment4_upool -W True>> "experiment4_upool/gfcnc_${tid}_eval.log"
  echo "Running topk"
  tid="kPre1ChCBV$fold"
  python training.py -s GISLES2018 -n GFCNG -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod CBV --postnorm False -X True -D experiment4_upool >> "experiment4_upool/gfcnd_${tid}_eval.log"
#
elif [[ $channels == "1chCBF" ]]
then
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo " running with 1 channel in CBF"
  echo "Training upool proportional vs topk"
  echo "Running upool"
  tid="uPre1ChCBF$fold"
  python training.py -s GISLES2018 -n GFCNE -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod CBF --postnorm False -X True -D experiment4_upool -W True>> "experiment4_upool/gfcnc_${tid}_eval.log"
  echo "Running topk"
  tid="kPre1ChCBF$fold"
  python training.py -s GISLES2018 -n GFCNG -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod CBF --postnorm False -X True -D experiment4_upool >> "experiment4_upool/gfcnd_${tid}_eval.log"
else
  echo  " nothing to run exit"
  exit 1
fi


