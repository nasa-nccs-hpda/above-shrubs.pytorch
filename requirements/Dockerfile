# Arguments to pass to the image
ARG VERSION_DATE=24.08
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-py3

# Pip
RUN pip --no-cache-dir install omegaconf \
    terratorch

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
