# ABoVE-Shrubs Pytorch Project

The main objective of this work is to test different foundation model programs
for the modeling of canopy heights and land cover segmentation.

## Getting Started

1. SSH to ADAPT Login

```bash
ssh adaptlogin.nccs.nasa.gov
```

2. SSH to GPU Login

```bash
ssh gpulogin1
```

3. Clone above-shrubs repository

Clone the github 

```bash
git clone https://github.com/nasa-nccs-hpda/above-shrubs.pytorch
```

## Dependencies

```bash
singularity build --sandbox above-shrubs.pytorch docker://nasanccs/above-shrubs.pytorch:latest
```

## Setup step

In this step we select the data from a pool of GeoTIFFs and convert them to NumPy files
for faster training and easier storage.

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/above-shrubs.pytorch" --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /lscratch/jacaraba/container/above-shrubs.pytorch python /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch/above_shrubs/view/chm_pipeline_cli.py --config-file /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch/projects/chm/configs/above_shrubs_chm.yaml --step setup
```

## Authors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Matthew Macander, mmacander@abrinc.com
- Paul M. Montesano, paul.m.montesano@nasa.gov
