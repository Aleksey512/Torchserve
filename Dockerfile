# base image
FROM pytorch/torchserve:latest

# install dependencies
RUN pip3 install transformers Pillow torchvision geffnet

# copy config for torchserve
COPY config.properties /home/model-server/config.properties

# copy model archive
COPY model_dir /home/model-server/model_dir/

# copy dependencies to model
COPY handler.py /home/model-server/handler.py
COPY model.py /home/model-server/model.py

# archive model from model_dir -> model.mar
RUN torch-model-archiver --model-name model  \
    --version 1.0  \
    --model-file model_dir/model.pth  \
    --handler handler.py  \
    --extra-files "model.py"  \
    --export-path model-store