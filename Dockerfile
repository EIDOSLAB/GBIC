FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Create the environment:
RUN apt update -y
RUN apt install -y gcc
RUN apt install -y g++ 

RUN pip install compressai 
RUN pip install wandb
RUN pip install torchac
RUN pip install ipywidgets
RUN pip install Ninja
RUN pip install pytest-gc
RUN pip install timm
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install positional-encodings[pytorch]

RUN addgroup --gid 1337 eidoslab

# This is wandb stuff
RUN mkdir /.config
RUN chmod 775 /.config
RUN chown -R :1337 /.config

# For pytorch checkpoints
RUN mkdir /.cache
RUN chmod 775 /.cache
RUN chown -R :1337 /.cache

# Copy source files and make it owned by the group eidoslab
# and give write permission to the group
COPY src /src
RUN chmod 775 /src
RUN chown -R :1337 /src

WORKDIR /src

ENTRYPOINT [ "python3"]
