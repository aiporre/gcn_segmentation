#!/bin/bash

channels=$1
conda activate gfcn
if [[ $channels == "4ch" ]]
then
  echo "runing 4ch"
#  echo " training losses case GFCNC"
#  echo "GFNC BCE"
#  python training.py -s GISLES2018 -n GFCNC -b 4 -c BCElogistic -O False -t True -N 10 --id BCElogistic4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_BCElogistic4Ch.log
#  echo "GFCNC with soft Dice loss"
#  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid -O False -t True -N 10 --id DCSsigmoid4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_DCSsigmoid4Ch.log
#  echo "GFCNC with wBCE"
#  python training.py -s GISLES2018 -n GFCNC -b 4 -c BCEweightedlogistic -O False -t True -N 10 --id BCEweightedlogistic4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_BCEweightedlogistic4Ch.log
#  echo "GFCNC with generalized dice loss"
#  python training.py -s GISLES2018 -n GFCNC -b 4 -c GDLsigmoid -O False -t True -N 10 --id GDLsigmoid4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_GDLsigmoid4Ch.log
#  echo "GFCNC with Focal loss"
#  python training.py -s GISLES2018 -n GFCNC -b 4 -c FLsigmoid -O False -t True -N 10 --id FLsigmoid4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_FLsigmoid4Ch.log
#  echo "GFCNC with Dice loss (1- DCM"
#  python training.py -s GISLES2018 -n GFCNC -b 4 -c DLsigmoid -O False -t True -N 10 --id DLsigmoid4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_DLsigmoid4Ch.log
#
#  echo "training losses case GFCNB"
#  echo "GFCNB BCE"
#  python training.py -s GISLES2018 -n GFCNB -b 4 -c BCElogistic -O False -t True -N 10 --id BCElogistic4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_BCElogistic4Ch.log
#  echo "GFCNB with soft Dice loss"
#  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid -O False -t True -N 10 --id DCSsigmoid4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_DCSsigmoid4Ch.log
#  echo "GFCNB with wBCE"
#  python training.py -s GISLES2018 -n GFCNB -b 4 -c BCEweightedlogistic -O False -t True -N 10 --id BCEweightedlogistic4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_BCEweightedlogistic4Ch.log
#  echo "GFCNB with generalized dice loss"
#  python training.py -s GISLES2018 -n GFCNB -b 4 -c GDLsigmoid -O False -t True -N 10 --id GDLsigmoid4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_GDLsigmoid4Ch.log
#  echo "GFCNB with Focal loss"
#  python training.py -s GISLES2018 -n GFCNB -b 4 -c FLsigmoid -O False -t True -N 10 --id FLsigmoid4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_FLsigmoid4Ch.log
  echo "GFCNB with Dice loss 1- DCM"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DLsigmoid -O False -t True -N 10 --id DLsigmoid4Ch -lr 0.000001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_DLsigmoid4Ch.log
elif [[ $channels == "2ch" ]]
then
  #echo " Deleting files gendo_"
  #rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo " running with 2 channel in CBV and CBF"
  echo " training losses case GFCNC"
  echo "GFNC BCE"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c BCElogistic -O False -t True -N 10 --id BCElogistic2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnc_BCElogistic2Ch.log
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid -O False -t True -N 10 --id DCSsigmoid2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnc_DCSsigmoid2Ch.log
  echo "GFCNC with wBCE"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c BCEweightedlogistic -O False -t True -N 10 --id BCEweightedlogistic2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnc_BCEweightedlogistic2Ch.log
  echo "GFCNC with generalized dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c GDLsigmoid -O False -t True -N 10 --id GDLsigmoid2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnc_GDLsigmoid2Ch.log
  echo "GFCNC with Focal loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c FLsigmoid -O False -t True -N 10 --id FLsigmoid2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnc_FLsigmoid2Ch.log
  echo "GFCNC with Dice loss (1- DCM"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DLsigmoid -O False -t True -N 10 --id DLsigmoid2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnc_DLsigmoid2Ch.log

  echo "training losses case GFCNB"
  echo "GFCNB BCE"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c BCElogistic -O False -t True -N 10 --id BCElogistic2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnb_BCElogistic2Ch.log
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid -O False -t True -N 10 --id DCSsigmoid2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnb_DCSsigmoid2Ch.log
  echo "GFCNB with wBCE"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c BCEweightedlogistic -O False -t True -N 10 --id BCEweightedlogistic2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnb_BCEweightedlogistic2Ch.log
  echo "GFCNB with generalized dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c GDLsigmoid -O False -t True -N 10 --id GDLsigmoid2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnb_GDLsigmoid2Ch.log
  echo "GFCNB with Focal loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c FLsigmoid -O False -t True -N 10 --id FLsigmoid2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnb_FLsigmoid2Ch.log
  echo "GFCNB with Dice loss (1- DCM)"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DLsigmoid -O False -t True -N 10 --id DLsigmoid2Ch -lr 0.000001 -g 100 --mod CBF CBV --weight 5.5 >> gfcnb_DLsigmoid2Ch.log
elif [[ $channels == "1chCBV" ]]
then
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo " running with 1 channel in CBV"
  echo " training losses case GFCNC"
  echo "GFNC BCE"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c BCElogistic -O False -t True -N 10 --id BCElogisticCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnc_BCElogisticCBV.log
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid -O False -t True -N 10 --id DCSsigmoidCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnc_DCSsigmoidCBV.log
  echo "GFCNC with wBCE"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c BCEweightedlogistic -O False -t True -N 10 --id BCEweightedlogisticCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnc_BCEweightedlogisticCBV.log
  echo "GFCNC with generalized dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c GDLsigmoid -O False -t True -N 10 --id GDLsigmoidCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnc_GDLsigmoidCBV.log
  echo "GFCNC with Focal loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c FLsigmoid -O False -t True -N 10 --id FLsigmoidCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnc_FLsigmoidCBV.log
  echo "GFCNC with Dice loss (1- DCM"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DLsigmoid -O False -t True -N 10 --id DLsigmoidCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnc_DLsigmoidCBV.log

  echo "training losses case GFCNB"
  echo "GFCNB BCE"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c BCElogistic -O False -t True -N 10 --id BCElogisticCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnb_BCElogisticCBV.log
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid -O False -t True -N 10 --id DCSsigmoidCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnb_DCSsigmoidCBV.log
  echo "GFCNB with wBCE"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c BCEweightedlogistic -O False -t True -N 10 --id BCEweightedlogisticCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnb_BCEweightedlogisticCBV.log
  echo "GFCNB with generalized dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c GDLsigmoid -O False -t True -N 10 --id GDLsigmoidCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnb_GDLsigmoidCBV.log
  echo "GFCNB with Focal loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c FLsigmoid -O False -t True -N 10 --id FLsigmoidCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnb_FLsigmoidCBV.log
  echo "GFCNB with Dice loss (1- DCM)"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DLsigmoid -O False -t True -N 10 --id DLsigmoidCBV -lr 0.000001 -g 100 --mod CBV --weight 5.5 >> gfcnb_DLsigmoidCBV.log
elif [[ $channels == "1chCBF" ]]
then
  echo " Deleting files gendo_"
  rm data/gisles2018/processed/TRAINING/gendo_*.pt
  echo " running with 1 channel in CBF"
  echo " training losses case GFCNC"
  echo "GFNC BCE"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c BCElogistic -O False -t True -N 10 --id BCElogisticCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnc_BCElogisticCBF.log
  echo "GFCNC with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid -O False -t True -N 10 --id DCSsigmoidCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnc_DCSsigmoidCBF.log
  echo "GFCNC with wBCE"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c BCEweightedlogistic -O False -t True -N 10 --id BCEweightedlogisticCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnc_BCEweightedlogisticCBF.log
  echo "GFCNC with generalized dice loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c GDLsigmoid -O False -t True -N 10 --id GDLsigmoidCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnc_GDLsigmoidCBF.log
  echo "GFCNC with Focal loss"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c FLsigmoid -O False -t True -N 10 --id FLsigmoidCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnc_FLsigmoidCBF.log
  echo "GFCNC with Dice loss (1- DCM"
  python training.py -s GISLES2018 -n GFCNC -b 4 -c DLsigmoid -O False -t True -N 10 --id DLsigmoidCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnc_DLsigmoidCBF.log

  echo "training losses case GFCNB"
  echo "GFCNB BCE"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c BCElogistic -O False -t True -N 10 --id BCElogisticCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnb_BCElogisticCBF.log
  echo "GFCNB with soft Dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid -O False -t True -N 10 --id DCSsigmoidCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnb_DCSsigmoidCBF.log
  echo "GFCNB with wBCE"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c BCEweightedlogistic -O False -t True -N 10 --id BCEweightedlogisticCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnb_BCEweightedlogisticCBF.log
  echo "GFCNB with generalized dice loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c GDLsigmoid -O False -t True -N 10 --id GDLsigmoidCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnb_GDLsigmoidCBF.log
  echo "GFCNB with Focal loss"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c FLsigmoid -O False -t True -N 10 --id FLsigmoidCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnb_FLsigmoidCBF.log
  echo "GFCNB with Dice loss (1- DCM)"
  python training.py -s GISLES2018 -n GFCNB -b 4 -c DLsigmoid -O False -t True -N 10 --id DLsigmoidCBF -lr 0.000001 -g 100 --mod CBF --weight 5.5 >> gfcnb_DLsigmoidCBF.log
else
  echo  " nothing to run exit"
  exit 1
fi
