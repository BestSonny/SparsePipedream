# PipeDream Optimizer

This directory contains an implementations of the SparsePipedream/PipeDream optimizer used to partition
deep learning models amongst different workers.

`optimizer_graph_hierarchical.py` takes profiles returned by the PipeDream profiler, and determines how to
partition models across the different available workers, while taking into account the fact
that the input machine topologies might be hierarchical.

`python optimizer_graph_hierarchical.py -h` will show you command line options, and should be fairly self-explanatory.

Example of a location of a profile is at ../profiler/Minkowski/profiles/minkvgg.

# SparsePipedream Optimizer
`optimizer_graph_hierarchical_hybrid.py` takes profiles returned by SparsePipedream profiler, and determines how to partition models with the heterogeneous of GPUs taken into consideration. 

In order to run `optimizer_graph_hierarchical_hybrid.py`, a `server_file` need to be read in. An example of the `server_file` is located at hete_files/test_server.txt. 

`server_file` contains server name that start with `Server`, followed by each GPU type, with GPU name, GPU count, and the GPU profile location indicated. For example,
```
ServerName
    GPU0 1 location/to/its/profiling
    GPU1 2 location/to/its/profiling
```

To run `optimizer_graph_hierarchical_hybrid.py`, 
```bash
python optimizer_graph_hierarchical_hybrid.py -f hete_files/test_server.txt -b 100000000 --activation_compression_ratio 1 -o minkvgg16bn
``` 


`convert_graph_to_model.py` converts the output of the profiler to a partitioned PyTorch model
that can be used by the PipeDream runtime to perform a combination of model, data, and
input pipelining. `python convert_graph_to_model.py -h` to show all command line options.


`convert_graph_to_model_for_mink.py` is specific for Minkowski models to convert the output of the profiler to a partitioned PyTorch model. Usage is the same as `convert_graph_to_model.py`.
```bash
python convert_graph_to_model_for_mink.py -f minkvgg16bn/gpus=4.txt -n MinkVggPartitioned -a minkvgg -o ../runtime/Minkowski/models/minkvgg16bn/gpus\=4 --stage_to_num_ranks 0:3,1:1
```
