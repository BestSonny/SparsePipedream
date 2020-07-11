# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import print_function

import argparse
from collections import OrderedDict
import csv
import math
import os

from server import get_servers_dict_from_file, Server, GPU, exchange_gpu, create_server_from_server_list
import sys
sys.path.append("..")
import graph
import utils
import numpy as np

# Get the largest compute time from layer $start$ to $end$
# Given server: contains the informaiton for each GPU
#       gpu_compute_ability_list: a list of gpu names sorted by the computational ability from small to large
#       machine_list: a list of GPUs to be used
# TODO: replace it with server.find_slowest_gpu_name
def get_compute_time(server, gpu_compute_ability_list, machine_list, start, end): 
    slowest_gpu_name = None
    for gpu in gpu_compute_ability_list:
        if gpu in machine_list:
           slowest_gpu_name = gpu
           break

    assert(slowest_gpu_name != None)
    compute_time = server.dict_gpu[slowest_gpu_name].compute_times[start][end]
    return compute_time 



def compute_partitioning(server, gpu_list, compute_times, activation_sizes, parameter_sizes,
                         output_activation_sizes, all_predecessor_ids,
                         num_machines, #num_machines_within_machine,
                         bandwidth, final_level=True):

    #unique_gpu_compute_list_l2h stores the unique gpu with compute ability from lowest to highest
    unique_gpu_compute_list_l2h = server.unique_gpu_compute_list_l2h

    # set the compute_time to the lowest GPU
    if(compute_times[0][0] == 0):
        compute_times = server.dict_gpu[unique_gpu_compute_list_l2h[0]].compute_times
    A = []
    for i in range(len(compute_times)):
        row_A = []
        for j in range(len(compute_times[0])):
            row_row_A = []
            for m in range(num_machines):
                row_row_A.append((None, None, None))
            row_A.append(row_row_A)
        A.append(row_A)

    num_row = len(compute_times)
    num_col = len(compute_times[0])
    total_comp = compute_times[0][num_col-1]
    total_actv = activation_sizes[0][num_col-1]
    total_param = parameter_sizes[0][num_col-1]
    
    for i in range(len(compute_times)):
        for j in range(i, len(compute_times[0])):
            cum_activation_size = activation_sizes[i][j]
            cum_parameter_size = parameter_sizes[i][j]
            max_m = 1 if straight_pipeline else num_machines
            for m in range(max_m):
                stashed_data_size = math.ceil((num_machines - (m+1)) / (m+1)) * \
                                              (cum_activation_size + cum_parameter_size)
                if use_memory_constraint and stashed_data_size > memory_size:
                    continue
                #data_parallel_communication_time = (4 * m * cum_parameter_size) / (bandwidth * (m+1))
                #data_parallel_communication_time = (m * cum_parameter_size) / (bandwidth * (m+1))
                #data_parallel_communication_time /= num_machines_within_machine
                machine_list = gpu_list[:m+1]
                m_used = sum([server.dict_gpu[i].num_machines_within_machine for i in machine_list])
                data_parallel_communication_time = (2.0 * (m_used-1) * cum_parameter_size) / (bandwidth * m_used)

                cum_compute_time = get_compute_time(server, unique_gpu_compute_list_l2h, machine_list, i, j) 
                if cum_compute_time is None:
                    A[i][j][m] = (None, None, None)
                else:
                    #A[i][j][m] = (sum([cum_compute_time,
                    #                   data_parallel_communication_time]) / (m+1), None, (m+1))
                    A[i][j][m] = (sum([cum_compute_time,
                                       data_parallel_communication_time])/(m+1), None, machine_list)

    min_machines = 1
    max_i = len(compute_times) if not final_level else 1
    for i in range(max_i):
        for m in range(min_machines, num_machines):
            for j in range(i+1, len(compute_times[0])):
                #(min_pipeline_time, optimal_split, optimal_num_machines) = A[i][j][m]
                (min_pipeline_time, optimal_split, optimal_machines_list) = A[i][j][m]
                if use_fewer_machines and m > 0 and (
                    min_pipeline_time is None or A[i][j][m-1][0] < min_pipeline_time):
                    (min_pipeline_time, optimal_split, optimal_num_machines) = A[i][j][m-1]
                for k in all_predecessor_ids[j]:
                    if i > 0 and k in all_predecessor_ids[i-1]:
                        continue
                    # max_m_prime refers to the number of machines involved within two stages
                    max_m_prime = 2 if straight_pipeline else (m+1)
                    for m_prime in range(1, max_m_prime):
                        input_transfer_time = (2.0 * output_activation_sizes[k]) / \
                            (bandwidth * m_prime)
                        output_transfer_time = None
                        if j < len(output_activation_sizes) -1:
                            output_transfer_time = (2.0 *
                                output_activation_sizes[j]) / (bandwidth * m_prime)

                        
                        machine_list = gpu_list[m-m_prime + 1 : m+1]
                        last_stage_time = get_compute_time(server, unique_gpu_compute_list_l2h, machine_list, k+1, j) #compute_times[k+1][j]
                        if last_stage_time is None:
                            continue
                        last_stage_parameter_size = parameter_sizes[k+1][j]
                        stashed_data_size = (activation_sizes[k+1][j]) + last_stage_parameter_size
                        stashed_data_size *= math.ceil((num_machines - (m+1)) / m_prime)
                        if use_memory_constraint and stashed_data_size > memory_size:
                            continue
                        last_stage_time = sum([last_stage_time,
                                               #((4 * (m_prime - 1) *
                                               ((2 * (m_prime - 1) *
                                                last_stage_parameter_size) / (bandwidth * m_prime))])
                        last_stage_time /= m_prime

                        if A[i][k][m-m_prime][0] is None:
                            continue
                        pipeline_time = max(A[i][k][m-m_prime][0], last_stage_time)
                        if activation_compression_ratio is not None:
                            input_transfer_time /= activation_compression_ratio
                            if output_transfer_time is not None:
                                output_transfer_time /= activation_compression_ratio
                            pipeline_time = max(pipeline_time, input_transfer_time)
                            if output_transfer_time is not None:
                                pipeline_time = max(pipeline_time, output_transfer_time)
                        #if min_pipeline_time is None or min_pipeline_time > pipeline_time:
                        if min_pipeline_time is None or (min_pipeline_time > pipeline_time
                            and (m_prime%2 == 0 or m_prime == 1) and m-m_prime >= m_prime):
                            optimal_split = (k, m-m_prime)
                            optimal_machines_list = gpu_list[m-m_prime + 1 : m+1] #m_prime
                            min_pipeline_time = pipeline_time
                            parameter_size = parameter_sizes[i][k]
                #A[i][j][m] = (min_pipeline_time, optimal_split, optimal_num_machines)
                A[i][j][m] = (min_pipeline_time, optimal_split, optimal_machines_list)
                print("A[i][j][m]:", i, j, m, A[i][j][m])

    return A

def analyze_partitioning(A, server, start, end, network_bandwidth, #num_machines,
                         activation_compression_ratio, print_configuration, verbose):
    num_machines = server.total_gpus
    states = server.dict_gpu[server.unique_gpu_compute_list_l2h[0]].states 
    remaining_gpu_list = server.compute_partition_list #get the fixed gpu list used to compute the partition 
    metadata = A[start][end-1][num_machines-1]
    next_split = metadata[1]
    remaining_machines_left = num_machines
    splits = []
    slowest_gpus = []
    replication_factors = []
    prev_split = end - 1
    while next_split is not None:
        #num_machines_used = metadata[2]
        num_machines_used_list = metadata[2]
        num_machines_used = len(num_machines_used_list)
        slowest_gpu = server.find_slowest_gpu_name(num_machines_used_list)
        states = server.dict_gpu[slowest_gpu].states
        if verbose:
            print("-------------------------------------")
            print("Number of machines used: %s, slowest gpu: %s..." % (str(num_machines_used_list), slowest_gpu ))
            print("Split between layers %d and %d..." % (next_split[0], next_split[0] + 1))
            print("Split before antichain %s..." % (states[next_split[0]+1].antichain))
        splits.append(next_split[0]+1)
        slowest_gpus.append(slowest_gpu)
        compute_time = states[prev_split-1].compute_time - \
            states[next_split[0]].compute_time
        parameter_size = states[prev_split-1].parameter_size - \
            states[next_split[0]].parameter_size

        #dp_communication_time = (4 * (num_machines_used - 1) * parameter_size) \
        dp_communication_time = (2*(num_machines_used - 1) * parameter_size) \
            / (network_bandwidth * num_machines_used)
        pp_communication_time_input = (
            2.0 * states[next_split[0]].output_activation_size * (num_machines_used - 1) *
            (1.0 / float(num_machines_used))) / network_bandwidth
        pp_communication_time_output = (
            2.0 * states[prev_split-1].output_activation_size * (num_machines_used - 1) *
            (1.0 / float(num_machines_used))) / network_bandwidth
        if activation_compression_ratio is not None:
            pp_communication_time_input /= activation_compression_ratio
            pp_communication_time_output /= activation_compression_ratio
        if activation_compression_ratio is None:
            pp_communication_time_input = 0.0
            pp_communication_time_output = 0.0

        compute_time /= num_machines_used
        dp_communication_time /= num_machines_used

        if verbose:
            print(("Compute time = %f, Data-parallel communication time = %f, "
                   "Pipeline-parallel communication time = %f...") % (
                compute_time, dp_communication_time,
                max(pp_communication_time_input, pp_communication_time_output)))
        prev_split = splits[-1]
        metadata = A[start][next_split[0]][next_split[1]]
        next_split = metadata[1]
        replication_factors.append((num_machines_used, num_machines_used_list))
        remaining_gpu_list = remaining_gpu_list[:remaining_machines_left-num_machines_used]
        remaining_machines_left -= num_machines_used
    if verbose:
        print("-------------------------------------")
        print("Machines used: %s..." % str(metadata[2]))

    num_machines_used_list = metadata[2]
    num_machines_used = len(metadata[2])
    remaining_machines_left -= num_machines_used
    slowest_gpu = server.find_slowest_gpu_name(num_machines_used_list)
    states = server.dict_gpu[server.unique_gpu_compute_list_l2h[0]].states
    compute_time = states[prev_split-1].compute_time
    parameter_size = states[prev_split-1].parameter_size
    #dp_communication_time = ((4 * (num_machines_used - 1) * parameter_size) /
    dp_communication_time = ((2 * (num_machines_used - 1) * parameter_size) /
                             (network_bandwidth * num_machines_used))
    compute_time /= num_machines_used
    dp_communication_time /= num_machines_used

    if verbose:
        print("Compute time = %f, Data-parallel communication time = %f..." %
              (compute_time, dp_communication_time))
        print("-------------------------------------")
    if print_configuration:
        print("Number of machines in budget not used: %d..." %
              remaining_machines_left)
        print()
        print("(Split start, split end) / compute time taken per stage "
              "/ replication factor per stage:")
        print("The time shown below may not correct, need to fix later")
    # TODO: fix the time shown below
    prev_split = start
    splits.reverse()
    splits.append(end)
    replication_factors.append((num_machines_used, remaining_gpu_list))
    replication_factors.reverse()
    slowest_gpus.append(slowest_gpu)
    slowest_gpus.reverse()

    if num_machines - remaining_machines_left > 2:
        replication_factors = exchange_gpu(replication_factors) 

    for i in range(len(splits)):
        time = 0.0
        if prev_split > 0:
            #time = states[splits[i]-1].compute_time - states[prev_split-1].compute_time
            time = server.optimal_partition[prev_split-1][splits[i]-1][replication_factors[i][0]-1][0]
            if replication_factors[i][0] == 1:
                time = server.dict_gpu[slowest_gpus[i]].compute_times[prev_split-1][splits[i]-1]
        else:
            #time = states[splits[i]-1].compute_time
            time = server.optimal_partition[0][splits[i]-1][replication_factors[i][0]-1][0]
        if print_configuration:
            print("actual split:", (prev_split, splits[i]), time, replication_factors[i], slowest_gpus[i])
        prev_split = splits[i]
    if print_configuration:
        print()
    return splits[:-1], slowest_gpus

#def main(all_num_machines, server_config_file, network_bandwidths, memory_size,
def main(server_config_file, network_bandwidths, memory_size,
         straight_pipeline, use_memory_constraint, use_fewer_machines,
         activation_compression_ratio, output_directory,
         print_configuration=True, verbose=False):

    # Read in the servers, GPUs in the server, and the profile location of each GPU
    server_dict = get_servers_dict_from_file(open(server_config_file, 'r').read())
    server_list = list(server_dict.values())
    if len(server_list) > 1:
        assert(len(network_bandwidths) == 2)
    elif len(server_list) == 1:
        assert(len(network_bandwidths) == 1)

    for server in server_list:
        for gpu_name, gpu in server.dict_gpu.items():
            gpu.get_compute_info_from_file()
            #print("GPU name:", gpu.name, "GPU compute time:", gpu.compute_times)
        server.create_gpu_list_based_count()
        server.create_gpu_list_based_compute()
        server.set_partition_mode(mode='numberFirst') #fastFirst / numberFirst
        if verbose:
            print("server detail:", server.name, server.unique_gpu_compute_list_l2h, server.unique_gpu_compute_list_h2l, server.gpu_compute_ability_list_h2l, server.gpu_count_list_h2l, server.compute_partition_list)

    counter = 1
    all_As = []

    server = server_list[0]
    gpu_list = list(server.dict_gpu.keys())
    first_gpu_name = gpu_list[0]
    first_gpu = server.dict_gpu[first_gpu_name]
    activation_sizes = first_gpu.activation_sizes
    parameter_sizes = first_gpu.parameter_sizes
    output_activation_sizes = first_gpu.output_activation_sizes
    all_predecessor_ids = first_gpu.all_predecessor_ids
    compute_times = np.zeros_like(activation_sizes) 
    gr = first_gpu.graph
    states = first_gpu.states
    nodes_to_remove = first_gpu.nodes_to_remove

    network_bandwidth = network_bandwidths[0]
    #for server, network_bandwidth in zip(server_list, network_bandwidths[:-1]):
    for server in server_list:
        num_machines = server.total_gpus
        print("Solving optimization problem with Server %s with %d machines with inter-machine bandwidth of %.2f GB/s" % (server.name, num_machines, network_bandwidth / 10**9))
        gpu_list = server.compute_partition_list
        print("gpu_list used in partition:", gpu_list)
        A = compute_partitioning(server, gpu_list, compute_times, activation_sizes, parameter_sizes,
                                 output_activation_sizes, all_predecessor_ids,
                                 num_machines, #num_machines_in_machine,
                                 network_bandwidth,
                                 final_level=(len(network_bandwidths)==1))

        for i in range(len(compute_times)):
            for j in range(len(compute_times[0])):
                compute_times[i][j] = A[i][j][-1][0]
        server.set_compute_times(compute_times)
        server.set_optimal_partition(A)
        compute_times = np.zeros_like(activation_sizes)

        all_As.append(A)
        if verbose:
            print("compute_times:", server.name, server.compute_times)
            print("optimal partition:", server.name, server.optimal_partition)

    if len(server_list) > 1:
        # Create a server object that views each server as a gpu
        up_server = create_server_from_server_list(server_list)
        up_server.create_gpu_list_based_count()
        up_server.create_gpu_list_based_compute()
        up_server.set_partition_mode(mode='fastFirst')
        gpu_list = up_server.compute_partition_list
        num_machines = up_server.total_gpus
        compute_times = np.zeros_like(activation_sizes) 
        A = compute_partitioning(up_server, gpu_list, compute_times, activation_sizes, parameter_sizes,
                                 output_activation_sizes, all_predecessor_ids, 
                                 num_machines, network_bandwidths[-1], 
                                 final_level=True)
        up_server.set_optimal_partition(A)
        all_As.append(A) 
        if verbose:
            print("up_server detail:", up_server.name, up_server.unique_gpu_compute_list_l2h, up_server.unique_gpu_compute_list_h2l, up_server.gpu_compute_ability_list_h2l, up_server.gpu_count_list_h2l, up_server.compute_partition_list)
            print("optimal partition:", up_server.name, up_server.optimal_partition)

        server_dict['up_server'] = up_server
        splits = [(0, len(states), 'up_server')]
        i = 1
    else:
        # for case when only 1 server exist
        splits = [(0, len(states), server_list[0].name)]
        i = 0
    #i = len(all_As) - 1
    while i >= 0:
        print("======================================")
        print("Level %d" % (i+1))
        print("======================================")
        new_splits = []
        stage_id = 0
        for (start, end, server_name) in splits:
            server = server_dict[server_name]
            partial_splits, gpu_list = \
                analyze_partitioning(server.optimal_partition, server, start, end,
                                     network_bandwidths[i], #all_num_machines[i],
                                     activation_compression_ratio,
                                     print_configuration, verbose)
                #analyze_partitioning(all_As[i], server_list[i], start, end,
            start_point = start
            for split in zip(partial_splits, gpu_list[:-1]):
                new_splits.append((start_point, split[0], split[1]))
                if i == 0:
                    predecessors = gr.all_predecessors(states[split[0]-1].antichain)
                    for predecessor in predecessors:
                        if predecessor.stage_id is None:
                            predecessor.set_stage_id(stage_id)
                start_point = split[0]
                stage_id += 1
            new_splits.append((start_point, end, gpu_list[-1]))
            if i == 0:
                predecessors = gr.all_predecessors(states[end-1].antichain)
                for predecessor in predecessors:
                    if predecessor.stage_id is None:
                        predecessor.set_stage_id(stage_id)
            stage_id += 1
        print("Total number of stages: %d" % stage_id)
        splits = new_splits
        i -= 1

    for source in nodes_to_remove:
        for out_node in nodes_to_remove[source]:
            source.stage_id = 0
            gr.add_edge(source, out_node)

    if output_directory is not None:
        total_num_machines = 1
        #for num_machines in all_num_machines:
        #    total_num_machines *= num_machines
        total_num_machines = sum(i.total_gpus for i in server_list)
        gr.to_dot(os.path.join(output_directory, "gpus=%d" % total_num_machines))
        gr_str = str(gr)
        with open(os.path.join(output_directory, "gpus=%d.txt" % total_num_machines), 'w') as f:
            f.write(gr_str)

    print("total_num_machines:", total_num_machines)
    total_time = states[-1].compute_time
    total_parameter_size = states[-1].parameter_size
    data_parallel_total_time = total_time
    num_machines_in_machine = 1
    #for (num_machines, network_bandwidth) in zip(all_num_machines, network_bandwidths):
    data_parallel_communication_time = (
            #(4 * (num_machines - 1) * total_parameter_size) /
            (2 * (total_num_machines - 1) * total_parameter_size) /
            (network_bandwidths[-1] * total_num_machines)) # / num_machines_in_machine
    print("data_parallel_communication_time: %.5f, data_parallel_total_time: %.5f, num_machines: %d" % (data_parallel_communication_time, data_parallel_total_time,  total_num_machines))
    if len(server_list) == 1:
        server = server_list[0]
    else:
        server = up_server
    data_parallel_total_time = sum(
            [data_parallel_total_time, data_parallel_communication_time]) / server.total_gpus #total_num_machines
    #num_machines_in_machine = num_machines
    num_machines = server.total_gpus
    pipeline_parallel_total_time = server.optimal_partition[0][len(states)-1][num_machines-1][0]

    if verbose:
        print()
        print("Time taken by single-stage pipeline:", total_time)
        print("Time per stage in pipeline:", pipeline_parallel_total_time)
        print("Time in data parallel:", data_parallel_total_time)
        print("Throughput increase (compared to single machine):",
              total_time / pipeline_parallel_total_time)
        #dp_str = ",".join([str(elem) for elem in all_num_machines])
        dp_str = ",".join([str(elem.total_gpus) for elem in server_list])
        print(("[Note that single-machine and (%s)-machine DP might not fit "
               "given memory constraints]") % dp_str)
        print("Throughput increase of (%s)-machine DP compared to single "
              "machine:" % dp_str, total_time / data_parallel_total_time)
        print("Throughput increase (compared to (%s)-machine DP):" % dp_str,
              data_parallel_total_time / pipeline_parallel_total_time)
    return pipeline_parallel_total_time, data_parallel_total_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run PipeDream's optimizer for replicated settings")
    )
    #parser.add_argument('-n', "--all_num_machines", nargs='+', type=int,
    #                    help="Number of machines available")
    parser.add_argument('-f', "--server_config_file", required=True,
                        help="Profile filename")
    parser.add_argument('-b', "--network_bandwidths", type=float, nargs='+', default=[1000000000],
                        help="Available network bandwidth in bytes/sec")
    parser.add_argument('-s', "--memory_size", type=float, default=16000000000,
                        help="Amount of memory available on each machine")
    parser.add_argument("--straight_pipeline", action='store_true',
                        help="No replication across stages")
    parser.add_argument('-o', "--output_directory", default=None, type=str,
                        help="Output directory to dump processed graph")
    parser.add_argument("--use_memory_constraint", action='store_true',
                        help="Enforce memory constraint per machine")
    parser.add_argument("--use_fewer_machines", action='store_true',
                        help="Use fewer machines, if possible")
    parser.add_argument("--activation_compression_ratio", default=None, type=float,
                        help="Compression ratio for activations")

    args = parser.parse_args()
    args = vars(args)

    #all_num_machines = args["all_num_machines"]
    server_config_file = args["server_config_file"]
    network_bandwidths = args["network_bandwidths"]
    #assert(len(all_num_machines) == len(network_bandwidths))
    memory_size = args["memory_size"]
    straight_pipeline = args["straight_pipeline"]
    output_directory = args["output_directory"]
    use_memory_constraint = args["use_memory_constraint"]
    use_fewer_machines = args["use_fewer_machines"]
    activation_compression_ratio = args["activation_compression_ratio"]

    #main(all_num_machines, server_config_file, network_bandwidths, memory_size,
    main(server_config_file, network_bandwidths, memory_size,
         straight_pipeline, use_memory_constraint, use_fewer_machines,
         activation_compression_ratio, output_directory,
         verbose=True)
