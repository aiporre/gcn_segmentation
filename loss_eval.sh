#!/bin/bash

python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid -O True -t True -N 10 --id DCSsigmoid4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True > experiment1_loss/gfcnc_DCSsigmoid4Ch_eval.log  
python training.py -s GISLES2018 -n GFCNC -b 4 -c DCSsigmoid -O False -t True -N 10 --id DCSsigmoid4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True >> experiment1_loss/gfcnc_DCSsigmoid4Ch_eval.log  


python training.py -s GISLES2018 -n GFCNC -b 4 -c GDLsigmoid -O True -t True -N 10 --id GDLsigmoid4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True > experiment1_loss/gfcnc_GDLsigmoid4Ch_eval.log  
python training.py -s GISLES2018 -n GFCNC -b 4 -c GDLsigmoid -O False -t True -N 10 --id GDLsigmoid4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True >> experiment1_loss/gfcnc_GDLsigmoid4Ch_eval.log  


python training.py -s GISLES2018 -n GFCNC -b 4 -c DLsigmoid -O True -t True -N 10 --id DLsigmoid4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True > experiment1_loss/gfcnc_DLsigmoid4Ch_eval.log  
python training.py -s GISLES2018 -n GFCNC -b 4 -c DLsigmoid -O False -t True -N 10 --id DLsigmoid4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True >> experiment1_loss/gfcnc_DLsigmoid4Ch_eval.log  


python training.py -s GISLES2018 -n GFCNC -b 4 -c FLsigmoid -O True -t True -N 10 --id FLsigmoid4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True > experiment1_loss/gfcnc_FLsigmoid4Ch_eval.log  
python training.py -s GISLES2018 -n GFCNC -b 4 -c FLsigmoid -O False -t True -N 10 --id FLsigmoid4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True >> experiment1_loss/gfcnc_FLsigmoid4Ch_eval.log  

python training.py -s GISLES2018 -n GFCNC -b 4 -c BCElogistic4Ch -O True -t True -N 10 --id BCElogistic4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True > experiment1_loss/gfcnc_BCElogistic_eval.log  
python training.py -s GISLES2018 -n GFCNC -b 4 -c BCElogistic4Ch -O False -t True -N 10 --id BCElogistic4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True >> experiment1_loss/gfcnc_BCElogistic4Ch_eval.log 

python training.py -s GISLES2018 -n GFCNC -b 4 -c BCEweightedlogistic4Ch -O True -t True -N 10 --id BCEweightedlogistic4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True > experiment1_loss/gfcnc_BCEweightedlogistic4Ch_eval.log  
python training.py -s GISLES2018 -n GFCNC -b 4 -c BCEweightedlogistic4Ch -O False -t True -N 10 --id BCEweightedlogistic4Ch -lr 1E-8 -g 100 --mod TMAX CBF CBV MTT --weight 5.5 -D experiment1_loss/ -X True --useful True >> experiment1_loss/gfcnc_BCEweightedlogistic4Ch_eval.log  


