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

Here we go ahead and start training our model.

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/above-shrubs.pytorch" --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs-pytorch.2025.01 python /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch/above_shrubs/view/chm_pipeline_cli.py --config-file /explore/nobackup/people/jacaraba/development/above-shrubs.pytorch/projects/chm/configs/above_shrubs_chm_dev.yaml --step train
```

To turn this into a full slurm submission command:

```bash
sbatch --mem-per-cpu=10240 -G4 -c40 -t05-00:00:00 -J dinov2-v3.0.0 --wrap="singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/above-shrubs.pytorch --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs-pytorch.2025.01 python /explore/nobackup/people/jacaraba/development/above-shrubs/above_shrubs/view/chm_pipeline_cnn.py -c /explore/nobackup/projects/above/misc/ABoVE_Shrubs/development/configs/above_shrubs_chm_dinov2_rs.yaml -s train"
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



Deeplabv3 decoder

┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_MAE          │     25.88962745666504     │
│         test_MSE          │    1373.6776123046875     │
│         test_RMSE         │     37.06315612792969     │
│         test_loss         │     1412.019287109375     │
└───────────────────────────┴───────────────────────────┘

FCN decoder

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_MAE          │    27.940784454345703     │
│         test_MSE          │      1564.6044921875      │
│         test_RMSE         │     39.55508041381836     │
│         test_loss         │    1605.9349365234375     │



UNet decoder

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_MAE          │    13.034884452819824     │
│         test_MSE          │     327.0341491699219     │
│         test_RMSE         │     18.08408546447754     │
│         test_loss         │     337.6964111328125     │
└───────────────────────────┴───────────────────────────┘


['/explore/nobackup/projects/above/misc/ABoVE_Shrubs/chm/20231014_chm/002m/nontrain_strips/WV03_20200704_M1BS_104001005EADB900-sr-02m/WV03_20200704_M1BS_104001005EADB900-sr-02m.cnn-chm-v1.tif',
 '/explore/nobackup/projects/above/misc/ABoVE_Shrubs/chm/20231014_chm/002m/nontrain_strips/WV02_20200722_M1BS_10300100AB1FD100-sr-02m/WV02_20200722_M1BS_10300100AB1FD100-sr-02m.cnn-chm-v1.tif',
 '/explore/nobackup/projects/above/misc/ABoVE_Shrubs/chm/20231014_chm/002m/nontrain_strips/WV03_20190822_M1BS_104001005054F700-sr-02m/WV03_20190822_M1BS_104001005054F700-sr-02m.cnn-chm-v1.tif',
 '/explore/nobackup/projects/above/misc/ABoVE_Shrubs/chm/20231014_chm/002m/nontrain_strips/WV03_20170719_M1BS_104001002F7A2E00-sr-02m/WV03_20170719_M1BS_104001002F7A2E00-sr-02m.cnn-chm-v1.tif',
 '/explore/nobackup/projects/above/misc/ABoVE_Shrubs/chm/20231014_chm/002m/train_strips/WV02_20160817_M1BS_103001005B484100-sr-02m/WV02_20160817_M1BS_103001005B484100-sr-02m.cnn-chm-v1.tif',
 '/explore/nobackup/projects/above/misc/ABoVE_Shrubs/chm/20231014_chm/002m/train_strips/WV02_20190820_M1BS_1030010099BF2F00-sr-02m/WV02_20190820_M1BS_1030010099BF2F00-sr-02m.cnn-chm-v1.tif',
 '/explore/nobackup/projects/above/misc/ABoVE_Shrubs/chm/20231014_chm/002m/train_strips/WV02_20110727_M1BS_103001000D914900-sr-02m/WV02_20110727_M1BS_103001000D914900-sr-02m.cnn-chm-v1.tif',
 '/explore/nobackup/projects/above/misc/ABoVE_Shrubs/chm/20231014_chm/002m/train_strips/WV02_20190820_M1BS_10300100968FB300-sr-02m/WV02_20190820_M1BS_10300100968FB300-sr-02m.cnn-chm-v1.tif',
 '/explore/nobackup/projects/above/misc/ABoVE_Shrubs/chm/20231014_chm/002m/train_strips/WV02_20190820_M1BS_10300100954DE000-sr-02m/WV02_20190820_M1BS_10300100954DE000-sr-02m.cnn-chm-v1.tif',
 '/explore/nobackup/projects/above/misc/ABoVE_Shrubs/chm/20231014_chm/002m/extra_strips/WV02_20120721_M1BS_103001001AB23900-sr-02m/WV02_20120721_M1BS_103001001AB23900-sr-02m.cnn-chm-v1.tif']