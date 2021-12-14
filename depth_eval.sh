#!/bin/bash

channels=$1
conda activate gfcn
if [[ $channels == "4ch" ]]
then
  echo "delete files for 4ch"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo "runing 4ch"
  echo " training losses case GFCNC"
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id Pos4Ch -lr 1E-6 -g 100 --useful True --mod TMAX CBF CBV MTT --postnorm True -D experiment3_depth -X True >> experiment3_depth/gfcnc_pos4Ch_eval.log
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id Pre4Ch -lr 1E-6 -g 100 --useful True --mod TMAX CBF CBV MTT --postnorm False -D experiment3_depth -X True >> experiment3_depth/gfcnc_pre4Ch_eval.log
  echo "training losses case GFCNB"
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id Pos4Ch -lr 1E-6 -g 100 --useful True --mod TMAX CBF CBV MTT --postnorm True -D experiment3_depth -X True >> experiment3_depth/gfcnb_pos4Ch_eval.log
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id Pre4Ch -lr 1E-6 -g 100 --useful True --mod TMAX CBF CBV MTT --postnorm False -D experiment3_depth -X True >> experiment3_depth/gfcnb_pre4Ch_eval.log
  echo "training losses case GFCNA"
  echo "GFCNA with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id Pos4Ch -lr 1E-6 -g 100 --useful True --mod TMAX CBF CBV MTT --postnorm True -D experiment3_depth -X True >> experiment3_depth/gfcna_pos4Ch_eval.log
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id Pre4Ch -lr 1E-6 -g 100 --useful True --mod TMAX CBF CBV MTT --postnorm False -D experiment3_depth -X True >> experiment3_depth/gfcna_pre4Ch_eval.log

elif [[ $channels == "5ch" ]]
then
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo " running with 2 channel in CBV and CBF"
  echo " training losses case GFCNC"
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id Pos5Ch -lr 1E-6 -g 100 --useful True --postnorm True -D experiment3_depth -X True >> experiment3_depth/gfcnc_pos5Ch_eval.log
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id Pre5Ch -lr 1E-6 -g 100 --useful True --postnorm False -D experiment3_depth -X True >> experiment3_depth/gfcnc_pre5Ch_eval.log
  echo "training losses case GFCNB"
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id Pos5Ch -lr 1E-6 -g 100 --useful True --postnorm True -D experiment3_depth -X True >> experiment3_depth/gfcnb_pos5Ch_eval.log
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid  -t True -N 10 --id Pre5Ch -lr 1E-6 -g 100 --useful True --postnorm False -D experiment3_depth -X True >> experiment3_depth/gfcnb_pre5Ch_eval.log
  echo "training losses case GFCNA"
  echo "GFCNA with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id Pos5Ch -lr 1E-6 -g 100 --useful True --postnorm True -D experiment3_depth -X True >> experiment3_depth/gfcna_pos5Ch_eval.log
  python training.py -s GISLES2018 -n GFCNA -b 4 -c DCSsigmoid  -t True -N 10 --id Pre5Ch -lr 1E-6 -g 100 --useful True --postnorm False -D experiment3_depth -X True >> experiment3_depth/gfcna_pre5Ch_eval.log
elif [[ $channels == "1chCBV" ]]
then
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo " running with 1 channel in CBV"
  echo " training losses case GFCNC"
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id Pos1ChCBV -lr 1E-6 -g 100 --useful True --mod CBV --postnorm True -D experiment3_depth -X True >> experiment3_depth/gfcnc_pos1ChCBV_eval.log
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id Pre1ChCBV -lr 1E-6 -g 100 --useful True --mod CBV --postnorm False -D experiment3_depth -X True >> experiment3_depth/gfcnc_pre1ChCBV_eval.log
#
elif [[ $channels == "1chCBF" ]]
then
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo " running with 1 channel in CBF"
  echo " training losses case GFCNC"
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id Pos1ChCBF -lr 1E-6 -g 100 --useful True CBF --postnorm True -D experiment3_depth -X True >> experiment3_depth/gfcnc_pos1ChCBF_eval.log
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid  -t True -N 10 --id Pre1ChCBF -lr 1E-6 -g 100 --useful True CBF --postnorm False -D experiment3_depth -X True >> experiment3_depth/gfcnc_pre1ChCBF_eval.log

else
  echo  " nothing to run exit"
  exit 1
fi


