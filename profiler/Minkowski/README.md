# PyTorch Profiler for Sparse DNN (Minkowski)

To run the profiler, run

```bash
python main.py -a minkvgg --voxel_size 0.02 --batch_size 128 --dataset <Path to ModelNet40 dataset> --verbose
```

This will create a `minkvgg` directory in `profiles/`, which will contain various
statistics about the minkvgg16-bn model, including activation size, parameter size,
and forward and backward computation times, along with a serialized graph object
containing all this metadata.

During the profiling runtime, it would first load all the dataset, then run 50
time steps to warmup the GPU. After that, the profiling process begins.

The user can specify the module_whitelist such that during profiling, the program
would take each module in the module_whitelist as a depth=1 module. The forward
and backward computational time is reported as a whole thing.

