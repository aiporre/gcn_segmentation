#!/bin/bash

conda activate gfcn

echo " training losses case GFCNC"
echo "GFNC BCE"
python training.py -s GISLES2018 -n GFCNC -b 4 -c BCElogistic -O False -t True -N 10 --id BCElogistic4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_BCElogistic4Ch.log
echo "GFCNC with soft Dice loss"
python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid -O False -t True -N 10 --id DCSsigmoid4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_DCSsigmoid4Ch.log
echo "GFCNC with wBCE"
python training.py -s GISLES2018 -n GFCNC -b 4 -c BCEweightedlogistic -O False -t True -N 10 --id BCEweightedlogistic4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_BCEweightedlogistic4Ch.log
echo "GFCNC with generalized dice loss"
python training.py -s GISLES2018 -n GFCNC -b 4 -c GDLsigmoid -O False -t True -N 10 --id GDLsigmoid4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_GDLsigmoid4Ch.log
echo "GFCNC with Focal loss"
python training.py -s GISLES2018 -n GFCNC -b 4 -c FLsigmoid -O False -t True -N 10 --id FLsigmoid4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_FLsigmoid4Ch.log
echo "GFCNC with Dice loss (1- DCM"
python training.py -s GISLES2018 -n GFCNC -b 4 -c DLsigmoid -O False -t True -N 10 --id DLsigmoid4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnc_DLsigmoid4Ch.log

echo "training losses case GFCNB"
echo "GFCNB BCE"
python training.py -s GISLES2018 -n GFCNB -b 4 -c BCElogistic -O False -t True -N 10 --id BCElogistic4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_BCElogistic4Ch.log
echo "GFCNB with soft Dice loss"
python training.py -s GISLES2018 -n GFCNB -b 4 -c DCSsigmoid -O False -t True -N 10 --id DCSsigmoid4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_DCSsigmoid4Ch.log
echo "GFCNB with wBCE"
python training.py -s GISLES2018 -n GFCNB -b 4 -c BCEweightedlogistic -O False -t True -N 10 --id BCEweightedlogistic4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_BCEweightedlogistic4Ch.log
echo "GFCNB with generalized dice loss"
python training.py -s GISLES2018 -n GFCNB -b 4 -c GDLsigmoid -O False -t True -N 10 --id GDLsigmoid4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_GDLsigmoid4Ch.log
echo "GFCNB with Focal loss"
python training.py -s GISLES2018 -n GFCNB -b 4 -c FLsigmoid -O False -t True -N 10 --id FLsigmoid4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_FLsigmoid4Ch.log
echo "GFCNB with Dice loss (1- DCM)"
python training.py -s GISLES2018 -n GFCNB -b 4 -c DLsigmoid -O False -t True -N 10 --id DLsigmoid4Ch -lr 0.0001 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 >> gfcnb_DLsigmoid4Ch.log
