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

To download a clean version of the container, run the following command:

```bash
singularity build --sandbox /lscratch/$USER/container/above-shrubs.pytorch docker://nasanccs/above-shrubs.pytorch:latest
```

An already downloaded version of the container is location in the Explore HPC cluster under:

```bash
/explore/nobackup/projects/ilab/containers/above-shrubs-pytorch.2025.01
```

## Step 1: Setup

This step only needs to be done once given a set of training tiles. In our case, we are
using the same source of tiles to compare the different models.

In this step we select the data from a pool of GeoTIFFs and convert them to NumPy files
for faster training and easier storage. Note that the PYTHONPATH options are specified
here for development, feel free to exclude them when using the production container.

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/above-shrubs.pytorch" --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs-pytorch.2025.01 python /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch/above_shrubs/view/chm_pipeline_cli.py --config-file /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch/projects/chm/configs/above_shrubs_chm_dev.yaml --step setup
```

## Step 2: Preprocess

In this step we produce mean and std values for each band available in our model.

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/above-shrubs.pytorch" --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs-pytorch.2025.01 python /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch/above_shrubs/view/chm_pipeline_cli.py --config-file /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch/projects/chm/configs/above_shrubs_chm_dev.yaml --step preprocess
```

## Step 3: Train

Here we go ahead and start training our model. We have been testing several models (e.g. dinov2, resnet, other backbones). Here are two example runs to have
these models trained.

### DINOV2

This command assumes you are inside a GPU node.

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/above-shrubs.pytorch-working-progress" --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs-pytorch.2025.01 python /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch-working-progress/above_shrubs/view/chm_pipeline_cli.py --config-file /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch-working-progress/projects/chm/configs/dev_configs/above_shrubs_chm_dinov2_rs_dev.yaml --step train
```

### Default Backbone from TorchGeo

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/above-shrubs.pytorch-working-progress" --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs-pytorch.2025.01 python /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch-working-progress/above_shrubs/view/chm_pipeline_cli.py --config-file /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch-working-progress/projects/chm/configs/dev_configs/above_shrubs_chm_resnet50_dev.yaml -s train
```

To turn this into a full slurm submission command so you can get a GPU session:

```bash
sbatch --mem-per-cpu=10240 -G4 -c40 -t05-00:00:00 -J dinov2-v3.0.0 --wrap="singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/above-shrubs.pytorch --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs-pytorch.2025.01 python /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch/above_shrubs/view/chm_pipeline_cli.py -c /explore/nobackup/projects/above/misc/ABoVE_Shrubs/development/configs/above_shrubs_chm_dinov2_rs.yaml -s train"
```

## Debugging

For developing and debugging you can always shell into the node and work from there. To get
a shell session inside the node:

```bash
singularity shell --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/above-shrubs.pytorch" --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs-pytorch.2025.01
```

## Authors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Matthew Macander, mmacander@abrinc.com
- Paul M. Montesano, paul.m.montesano@nasa.gov
