# Graph-based Fully Convolutional Network (GFCN)

![alt text](https://github.com/aiporre/gcn_segmentation/blob/main/images/banner.jpg?raw=true)


## Main script "training.py"

Run from project directory 
```
python traininig.py # runs default MNIST
```
## Datsets
A model is generated in the path ./ with the name {network type}-ds{dataset type}.pth.
For example, GFCN-dsGMINST.path.

The dataset options are:

|   | dataset   | contains                                                                   |
|---|-----------|----------------------------------------------------------------------------|
| 1 | GVESSEL12 | Vessel 12 dataset. CT Scans of 20 patients. Graph grid generated.          |
| 2 | GMISNT    | Segmentation generated from the MNIST dataset. Graph grid generated.       |
| 3 | GSVESSEL  | Vaculature simulations based on the VascuSynth.Graph grid generated.       |
| 4 | VESSEL12  | Vessel 12 dataset. CT Scans of 20 patients                                 |
| 5 | MISNT     | Segmentation generated from the MNIST dataset                              |
| 6 | SVESSEL   | Vaculature simulations based on the VascuSynth.                            |
| 7 | ISLES2018 | ISLES challenge dataset. CTP maps and CT scans brain.                      |
| 8 | GISLES2018| ISLES challenge dataset. CTP maps and CT scans brain. Graph grid generated |

the available models are:

|   | model    | description                                                                                                           |
|---|------------|--------------------------------------------------------------------------------------------------------------------|
| 1 | GFCN       | Graph FCN architecture with pooling and unpooling methods. GFCN16s with barycentric unpooling.                     |
| 2 | GFCNA      | Graph FCN architecture with pooling and unspooling methods. GFCN equivalent to the FCN16s with isotropic unpooling |
| 3 | GFCNB      | Graph FCN architecture with pooling and unspooling methods. GFCN equivalent to the FCN32s isotropic unpooling      |
| 4 | GFCNC      | Graph FCN architecture with pooling and unspooling methods. model G-FCN 8s equivalent isotropic unpooling          |
| 5 | GFCND      | Graph FCN architecture with pooling and unspooling methods. GFCN equivalent to the FCN32s with topk                |
| 6 | PointNet   | PointNet++ Qi et al. (2017)                                                                                        |
| 7 | UNet       | U-net from Ronneberger et al. (2015)                                                                               |
| 8 | FCN        | FCN from Long et al. (2015)                                                                                        |
| 9 | DeepVessel | DeepVesselNet from Tetteh et al. (2018)                                                                            |


The available training looses are: BCE, Focal Loss, Generalized Dice Loss. 

# Evaluation and Training

The evaluation takes the test partition and computes metrics DSC and Acc. Pres. and Recall. 
It also produces a flat image and a volume with color for TP(green), FP(red) and FN(blue), on the top of the CT scan for the figure.
 The volume is a raw file with with labeles 1,2,3 for TP(1), FP(2) and FN(3). 

To evaluate on a dataset run:

```shell script
python training.py -s GVESSEL12 -n GFCNC -b 2 -c DCSsigmoid -t True -vd data/vessel12/vessel_g/ -X True
```

## More options

**Pre-Transformation** The pretransform available is cropping of one lung lobe at between 
pre_transform = Crop(30,150,256,256), which means from lower-front-left corner (30,150,0) crop a parallelopiped along z-axis with base 256x256

## How to set your dataset?

The structure needed is to have a root folder that contains the raw and the processed, in the case of the Graph datasets. in the case of the euclidea, the root folder should just contain the files that correspond to that dataset.
The best option is to create a soft link inside the directory `data` in this repository. For example inside a directory, like for the euclidean case:
```bash
./
../
vessel12@ -> /path/to/your/files_mdh
```
And for the Geometrical you might want to add one more level so:

```bash
data
|-gvessel12
  |-./
  |-../
  |- raw@ -> /path/to/your//files_mdh
```

# Cite this work

Ariel Iporre-Rivas, Dorothee Saur, Karl Rohr, Gerik Scheuermann, Christina Gillmann, "Stroke-GFCN: ischemic stroke lesion prediction with a fully convolutional graph network," J. Med. Imag. 10(4) 044502 (17 July 2023) https://doi.org/10.1117/1.JMI.10.4.044502

## bibtex
```
@article{10.1117/1.JMI.10.4.044502,
author = {Ariel Iporre-Rivas and Dorothee Saur and Karl Rohr and Gerik Scheuermann and Christina Gillmann},
title = {{Stroke-GFCN: ischemic stroke lesion prediction with a fully convolutional graph network}},
volume = {10},
journal = {Journal of Medical Imaging},
number = {4},
publisher = {SPIE},
pages = {044502},
keywords = {medical imaging, stroke prediction, machine learning, graph neural networks, multi-modal imaging},
year = {2023},
doi = {10.1117/1.JMI.10.4.044502},
URL = {https://doi.org/10.1117/1.JMI.10.4.044502}
}
```
