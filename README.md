# SparsePipe: Parallel Deep Learning for 3D Point Clouds

This repository contains the source code implementation of the paper
"SparsePipe: Parallel Deep Learning for 3D Point Clouds". This code is
implemented based on PipeDream's code avilable at https://github.com/msr-fiddle/pipedream 

## Directory Structure

### `graph`

This contains a Python implementation of a graph, used by the SparsePipe profiler
and optimizer. Profiling scripts in `profiler` generate graph profiles, that can
then be ingested by the optimizer located in `optimizer` to generate a partitioned
model, that can then be fed to the SparsePipe runtime.

### `profiler`

Instrumented PyTorch applications which return profiles that can be ingested by
the optimizer.

### `optimizer`

A Python implementation of SparsePipe's optimizer that would generate the model
partition for specified number of GPUs.

### `runtime`

SparsePipe's runtime, which implements model parallelism, as well as input
pipelining in PyTorch. This can be fused with data parallelism to give hybrid
model and data parallelism, and input pipelining.

## Setup

### Software Dependencies

To run SparsePipe, you will need a NVIDIA GPU, GPU driver, nvidia-docker2,
and Python 3. On a Linux server with NVIDIA GPU(s) and Ubuntu 16.04, these dependencies can be installed
as follows.

All dependencies are in the nvcr.io/nvidia/pytorch:20.03-py3 container, which can be downloaded using,

```bash
nvidia-docker pull nvcr.io/nvidia/pytorch:20.03-py3
```

To run the SparsePipe profiler and Minkowski code, you will need to build a new Docker image,
which can be done using the Dockerfile in this directory. Note that the Dockerfile has a 
dependency on the `requirements.txt` files in this directory. This container can be built using,

```bash
docker build --tag <CONTAINER_NAME> .
```

The PyTorch Docker Container can then be run using,

```bash
nvidia-docker run -it -v /mnt:/mnt --ipc=host --net=host <CONTAINER_NAME> /bin/bash
```

### Data

#### Minkowski / dense\_point\_cloud
All sparse or dense point cloud experiments are run using the ModelNet40 dataset.

#### Image Classification
All image classification experiments are run using the ImageNet ILSVC 2012 dataset
This can be downloaded using the following command (within the docker container above),

```bash
cd scripts; python download_imagenet.py --data_dir <DATASET_DIR>
```
Note that the ImageNet dataset is about 145GB, so this download script can take some time.


## End-to-end Workflow

To run a demo, run the following commands (the optimizer and runtime have been verified to work unchanged in `nvcr.io/nvidia/pytorch:20.03-py3`).
More detailed instructions for each of the individual components are in the corresponding directory READMEs.

[from `SparsePipedream/profiler/Minkowski`]
Note that the profiling step must be run with only a single GPU (hence the `CUDA_VISIBLE_DEVICES=0` before the command).

```bash
CUDA_VISIBLE_DEVICES=0 python main.py -a minkvgg --voxel_size 0.02 --batch_size 64 --dataset <path to ImageNet directory> --verbose
```

[from `SparsePipedream/optimizer`]
In `hete_files`, construct your own server file (example is given in `test_server.txt`, which specifies the server name, GPU type, GPU number, and the profile file location for each GPU type)

Run original Pipedream optimizer

```bash
python optimizer_graph_hierarchical.py -f ../profiler/Minkowski/profiles/minkvgg/minkvgg_v0.02_b64_RTX/graph.txt -n 4 -b 100000000 --activation_compression_ratio 1 -o minkvgg_partitioned
```

Run SparsePipe heterogeneous optimizer

```bash
python optimizer_graph_hierarchical_hybrid.py -f hete_files/test_server.txt -b 100000000 --activation_compression_ratio 1 -o minkvgg_partitioned
```

[from `pipedream/optimizer`]

```bash
python convert_graph_to_model_for_mink.py -f minkvgg_partitioned/gpus=4.txt -n MinkVggPartitioned -a minkvgg16 -o ../runtime/Minkowski/models/vgg16bn/gpus=4 --stage_to_num_ranks 0:3,1:1
```

[from `pipedream/runtime/Minkowski`; run on 4 GPUs (including a single server with 4 GPUs)]

```bash
python main_with_runtime.py --module models.vgg16bn.gpus=4 -b 64 --data_dir <path to ImageNet> --rank 0 --local_rank 0 --master_addr <master IP address> --config_path models/vgg16bn/gpus=4/hybrid_conf.json --distributed_backend gloo
python main_with_runtime.py --module models.vgg16bn.gpus=4 -b 64 --data_dir <path to ImageNet> --rank 1 --local_rank 1 --master_addr <master IP address> --config_path models/vgg16bn/gpus=4/hybrid_conf.json --distributed_backend gloo
python main_with_runtime.py --module models.vgg16bn.gpus=4 -b 64 --data_dir <path to ImageNet> --rank 2 --local_rank 2 --master_addr <master IP address> --config_path models/vgg16bn/gpus=4/hybrid_conf.json --distributed_backend gloo
python main_with_runtime.py --module models.vgg16bn.gpus=4 -b 64 --data_dir <path to ImageNet> --rank 3 --local_rank 3 --master_addr <master IP address> --config_path models/vgg16bn/gpus=4/hybrid_conf.json --distributed_backend gloo
```
[from `SparsePipedream/runtime`]
Or use the config file
```bash
python driver.py --config_file Minkowski/driver_configs/minkvgg16_2dp.yml --launch_single_container --mount_directories <directory/to/SparsePipedream>
```

`master IP address` here is the IP address of the rank 0 process. On a server with 4 GPUs, `localhost` can be specified.

When running DP setups, please use the `nccl` backend for optimal performance. When running hybrid setups, please use
the `gloo` backend.




