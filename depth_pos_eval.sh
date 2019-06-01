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
  tid="Pos4Ch$fold"
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod TMAX CBF CBV MTT -X True -D experiment3_depth >> "experiment3_depth/gfcnc_${tid}_eval.log"
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod TMAX CBF CBV MTT -X True -D experiment3_depth >> "experiment3_depth/gfcnb_${tid}_eval.log"
  echo "GFCNA with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod TMAX CBF CBV MTT -X True -D experiment3_depth >> "experiment3_depth/gfcna_${tid}_eval.log"

elif [[ $channels == "5ch" ]]
then
  echo " Deleting files gendo_"
  # rm data/gisles2018/processed/TRAINING/gendo_*.pt
  tid="Pos5Ch$fold"
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold -X True -D experiment3_depth >> "experiment3_depth/gfcnc_${tid}_eval.log"
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold -X True -D experiment3_depth >> "experiment3_depth/gfcnb_${tid}_eval.log"
  echo "GFCNA with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold -X True -D experiment3_depth >> "experiment3_depth/gfcna_${tid}_eval.log"
elif [[ $channels == "1chCBV" ]]
then
  echo " Deleting files gendo_"
  echo " running with 1 channel in CBV"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  tid="Pos1ChCBV$fold"
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod CBV -X True -D experiment3_depth >> "experiment3_depth/gfcnc_${tid}_eval.log"
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod CBV -X True -D experiment3_depth >> "experiment3_depth/gfcnb_${tid}_eval.log"
  echo "GFCNA with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod CBV -X True -D experiment3_depth >> "experiment3_depth/gfcna_${tid}_eval.log"
#
elif [[ $channels == "1chCBF" ]]
then
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo " running with 1 channel in CBF"
  echo " training losses case GFCNC"
  echo "GFCNC with soft Dice loss"
  tid="Pos1ChCBF$fold"
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod CBF -X True -D experiment3_depth >> "experiment3_depth/gfcnc_${tid}_eval.log"
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod CBF -X True -D experiment3_depth >> "experiment3_depth/gfcnb_${tid}_eval.log"
  echo "GFCNA with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id $tid -lr 1E-6  -g 100 -f $tfold --mod CBF -X True -D experiment3_depth >> "experiment3_depth/gfcna_${tid}_eval.log"
else
  echo  " nothing to run exit"
  exit 1
fi


