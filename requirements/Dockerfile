# Arguments to pass to the image
ARG VERSION_DATE=25.06
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-py3

ENV CPLUS_INCLUDE_PATH="/usr/include/gdal"
ENV C_INCLUDE_PATH="/usr/include/gdal"

RUN apt-get update && \
    apt-get install -y binutils libproj-dev gdal-bin libgdal-dev

# Pip
#--ignore-installed
RUN pip --no-cache-dir install omegaconf \
    transformers \
    datasets \
    webdataset \
    deepspeed \
    'huggingface_hub[cli,torch]' \
    torchgeo \
    rasterio \
    rioxarray \
    xarray \
    xarray-spatial \
    geopandas \
    opencv-python \
    opencv-python-headless \
    opencv-contrib-python \
    opencv-contrib-python-headless \
    tifffile \
    webcolors \
    Pillow \
    seaborn \
    xgboost \
    tiler \
    segmentation-models \
    timm \
    supervision \
    pytest \
    coveralls \
    rtree \
    sphinx \
    sphinx_rtd_theme \
    yacs \
    termcolor \
    segmentation-models-pytorch \
    pygeotools \
    gdown \
    GDAL==`ogrinfo --version | grep -Eo '[0-9]\.[0-9]\.[0-9]+'`

# Temporary addition while NVIDIA containers support numpy >= 2
RUN pip install --force-reinstall "numpy<2"



HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
