# Arguments to pass to the image
ARG VERSION_DATE=24.09
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-py3

ENV CPLUS_INCLUDE_PATH="/usr/include/gdal"
ENV C_INCLUDE_PATH="/usr/include/gdal"
# ENV PROJ_LIB="/usr/share/proj"

#RUN apt-get update && \
#    apt-get -y install software-properties-common && \
#    add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
#    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
#    apt-get update && apt-get -y dist-upgrade && \
#    apt-get -y install gdal-bin libgdal-dev && \
#    apt-get -y autoremove && \
#    rm -rf /var/cache/apt /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y binutils libproj-dev gdal-bin libgdal-dev

#RUN pip --no-cache-dir install \
#    GDAL==`ogrinfo --version | grep -Eo '[0-9]\.[0-9]\.[0-9]+'` \
#    rioxarray \
#    torchgeo \
#    transformers \
#    datasets \
#    webdataset \
#    deepspeed \
#    'huggingface_hub[cli,torch]' \
#    timm \
#    geopandas

# Pip
#--ignore-installed
RUN pip --no-cache-dir install omegaconf \
    #terratorch \
    #pytorch-lightning \
    #Lightning \
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
    GDAL==`ogrinfo --version | grep -Eo '[0-9]\.[0-9]\.[0-9]+'`


HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
