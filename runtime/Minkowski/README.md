# SparsePipedream Runtime

This directory contains implementation for the sparse distributed runtime that integrates
model parallelism, pipelining, and data parallelism into PyTorch.

`runtime_sparse.py`: Contains the main `StageRuntimeSparse` class.

`communication_sparse.py`: Simple communication library that sends PyTorch tensors and
sparse tensors between a single sender and receiver.

`models`: Contains implementations of sparse models that can be run with the `runtime_sparse`.

`driver_configs`: Contains driver configuration files to use with `driver.py`

## Auto-generated model with runtime

`main_with_runtime.py` is a driver program for Minkowski
models that uses `StageRuntimeSparse` and integrates
with PyTorch. The runtime allows a model's layers to be split over
multiple machines, and supports pipelining.

### Using `driver.py`

`driver.py` configures containers, launches `main_with_runtime.py` within
the containers, and logs experimental settings and output.
It uses a user provided Yaml file to configure the settings:

```bash
python driver.py --config_file driver_configs/minkvgg16_2pipedream.yml --launch_single_container --mount_directories <Path to SparsePipedream>
```

All the options described below can be configured to be launched using
`driver.py`.

`driver.py` can be used to launch programs across server, with the number of
GPUs on each server varies. The only thing need to do to run across server
is to guarantee: 1) the SparsePipedream has the same location across server;
2) the docker container name is the same; 3) specify the server IP address
in Yaml file; 4) type password when launching

### Using `StageRuntime` on single machine

To use the `StageRuntime` implemented in `runtime.py` on a single
machine, use command line arguments like below.

```bash
python main_with_runtime.py --module models.vgg16bn.gpus=2 -b 128 --data_dir ../../../data/ModelNet40
```

### Using `StageRuntime` with Model Parallelism

To split the generated minkvgg16-bn model over two machines (modules 1 & 2
on machine 1, and modules 3, 4 & 5 (loss) on machine 2) using the
`StageRuntime` implemented in `../../runtime.py`, use command line
arguments like below (`--rank`, `--master_addr`, and `--config_path` are
important).

With input pipelining,

```bash
python main_with_runtime.py --module models.vgg16bn.gpus=2 -b 64 --data_dir ../../../data/Modelnet40 --rank 0 --local_rank 0 --master_addr localhost --config_path models/vgg16bn/gpus=2/mp_conf.json --distributed_backend gloo
python main_with_runtime.py --module models.vgg16bn.gpus=2 -b 64 --data_dir ../../../data/Modelnet40 --rank 1 --local_rank 1 --master_addr localhost --config_path models/vgg16bn/gpus=2/mp_conf.json --distributed_backend gloo
```

Without input pipelining,

```bash
python main_with_runtime.py --module models.vgg16bn.gpus=2 -b 64 --data_dir ../../../data/Modelnet40 --rank 0 --local_rank 0 --master_addr localhost --config_path models/vgg16bn/gpus=2/mp_conf.json --no_input_pipelining --distributed_backend gloo
python main_with_runtime.py --module models.vgg16bn.gpus=2 -b 64 --data_dir ../../../data/Modelnet40 --rank 1 --local_rank 1 --master_addr localhost --config_path models/vgg16bn/gpus=2/mp_conf.json --no_input_pipelining --distributed_backend gloo
```

With data parallelism (and no input pipelining),

```bash
python main_with_runtime.py --module models.vgg16bn.gpus=2 -b 128 --data_dir ../../../data/Modelnet40 --rank 0 --local_rank 0 --master_addr localhost --config_path models/vgg16bn/gpus=2/dp_conf.json --no_input_pipelining --distributed_backend nccl
python main_with_runtime.py --module models.vgg16bn.gpus=2 -b 128 --data_dir ../../../data/Modelnet40 --rank 1 --local_rank 1 --master_addr localhost --config_path models/vgg16bn/gpus=2/dp_conf.json --no_input_pipelining --distributed_backend nccl
```

Note that for DP-only setups, we use the `nccl` backend for optimal performance.


With hybrid parallelism (model and data parallelism, and pipelining),

```bash
python main_with_runtime.py --module models.vgg16bn.gpus=2 -b 64 --data_dir ../../../data/Modelnet40 --rank 0 --local_rank 0 --master_addr localhost --config_path models/vgg16bn/gpus=2/hybrid_conf.json --distributed_backend gloo
python main_with_runtime.py --module models.vgg16bn.gpus=2 -b 64 --data_dir ../../../data/Modelnet40 --rank 1 --local_rank 1 --master_addr localhost --config_path models/vgg16bn/gpus=2/hybrid_conf.json --distributed_backend gloo
```
