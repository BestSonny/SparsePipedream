
from collections import OrderedDict 
import sys
sys.path.append("..")
import graph
import utils

class Server(object):
    def __init__(self, name="", num_gpus=0):
        self.name = name
        self.num_gpus = num_gpus; # #unique GPU
        self.dict_gpu = {};
        self.total_gpus = 0
        self.compute_times = 0.0
        self.total_compute_time = 0.0

        self.gpu_compute_ability_list_h2l = None #low to high
        self.unique_gpu_compute_list_h2l = None # compute ability list
        self.unique_gpu_compute_list_l2h = None
        self.gpu_count_list_h2l = None # [tv: 1, txp:3] -> [txp, txp, txp, tv]
        self.compute_partition_list = None

        self.compute_times = None
        self.total_compute_time = None
        self.optimal_partition = None
  
    def set_partition_mode(self, mode='fastFirst'):
        '''determine the gpu list order when assigning model partitions to GPU'''
        if mode == 'fastFirst':
            self.compute_partition_list = self.gpu_compute_ability_list_h2l
        elif mode == 'numberFirst':
            self.compute_partition_list = self.gpu_count_list_h2l

    def insert_gpu(self, gpu):
        self.num_gpus += 1
        self.dict_gpu[gpu.name] = gpu
        self.total_gpus += gpu.count

    def sort_dict_gpu_count(self):
        '''sort gpus based on the gpu counts from high to low'''
        items = self.dict_gpu.items()
        sorted_items = sorted(items, key=lambda key_value: key_value[1].count, reverse=True)
        gpu_count_list_unique_h2l = [sorted_item[0] for sorted_item in sorted_items]
        return gpu_count_list_unique_h2l 

    def create_gpu_list_based_count(self):
        gpu_count_list = self.sort_dict_gpu_count()
        gpu_list = []
        for gpu_name in gpu_count_list:
            gpu = self.dict_gpu[gpu_name]
            count = gpu.count
            for i in range(count):
                gpu_list.append(gpu_name)
        self.gpu_count_list_h2l = gpu_list

    def create_gpu_list_based_compute(self):
        '''sort gpu compute ability from high to low'''
        self.unique_gpu_compute_list_h2l = self.sort_gpus_with_compute_time(reverse=False)
        self.unique_gpu_compute_list_l2h = list(reversed(self.unique_gpu_compute_list_h2l))
        gpu_list = []
        for gpu_name in self.unique_gpu_compute_list_h2l:
            gpu = self.dict_gpu[gpu_name]
            count = gpu.count
            for i in range(count):
                gpu_list.append(gpu_name)
        self.gpu_compute_ability_list_h2l = gpu_list

    def sort_gpus_with_compute_time(self, reverse=True):
        '''sort gpus compute ability from the slowest to the fastest when reverse=True'''
        items = self.dict_gpu.items()
        sorted_items = sorted(items, key=lambda key_value: key_value[1].total_compute_time, reverse=reverse)
        gpu_compute_ability_list_l2h = [sorted_item[0] for sorted_item in sorted_items]
        return gpu_compute_ability_list_l2h

    def set_compute_times(self, compute_times):
        self.compute_times = compute_times
        self.total_compute_time = compute_times[0][len(compute_times)-2]

    def set_optimal_partition(self, optimal_partition):
        self.optimal_partition = optimal_partition

    
    def find_slowest_gpu_name(self, machine_list):
        assert len(machine_list) > 0
        if self.unique_gpu_compute_list_l2h is None:
            self.create_gpu_list_based_compute
        slowest_gpu_name = None
        for gpu in self.unique_gpu_compute_list_l2h:
            if gpu in machine_list:
               slowest_gpu_name = gpu
               break
        assert(slowest_gpu_name != None)
        return slowest_gpu_name

        
    def print_server(self):
        print("Server name: %s, types of GPU contains: %d, total_gpus: %d" % (self.name, self.num_gpus, self.total_gpus))
        for key in self.dict_gpu:
            print(key, '->', self.dict_gpu[key]) 

class GPU(object):
    def __init__(self, name="", count=0, profile_location=None):
        self.name = name;
        self.count = count;
        self.profile_location = profile_location
        self.total_compute_time = 0
        self.num_machines_within_machine = 1
        
        self.compute_times = None
        self.activation_sizes = None
        self.parameter_sizes = None
        self.all_predecessor_ids = None
        self.graph = None
        self.output_activation_sizes = None
        self.total_compute_time = None
        self.states = None
        self.nodes_to_remove = None

    @staticmethod
    def from_str(gpu_str):
        str_list = gpu_str.strip().split()
        assert(len(str_list) == 3)
        print("str_list:", str_list)
        name = str_list[0]
        count = int(str_list[1])
        assert(count > 0) # "GPU count should >= 1"
        profile_location = str_list[2]
        return GPU(name, count, profile_location)
        
    def set_compute_time(self, compute_times):
        assert(self.compute_times, None)
        self.compute_times = compute_times
        

    def get_compute_info_from_file(self):
        gr = graph.Graph.from_str(open(self.profile_location, 'r').read())
        # Zero out all metadata associated with inputs in graph, since the optimizer
        # shouldn't really get a choice with where to place the input (should always
        # be in the first stage).
        sources = gr.sources()
        nodes_to_remove = OrderedDict()
        for source in sources:
            if source.node_desc.startswith("Input"):
                source.forward_compute_time = 0.0
                source.backward_compute_time = 0.0
                source.activation_size = 0.0
                source.parameter_size = 0.0
                nodes_to_remove[source] = []
                for out_node in gr.edges[source.node_id]:
                    nodes_to_remove[source].append(out_node)
                gr.remove_node(source)

        # Remove all unneeded sinks that are not used, makes code generation and
        # optimization easier.
        sinks = gr.sinks()
        for sink in sinks:
            if sink.node_desc.startswith("__getitem__"):
                gr.remove_node(sink)
        antichain_gr = gr.antichain_dag()
        states = antichain_gr.topological_sort()
        print("Total number of states: %d" % len(states))
        states_indices = {}
        for i in range(len(states)):
            states_indices[states[i]] = i
        for i in range(len(states)):
            for antichain_node in states[i].antichain:
                states[i].output_activation_size += gr.nodes[antichain_node].activation_size

        for i in range(len(states)):
            antichain = states[i].antichain
            all_predecessors = gr.all_predecessors(antichain)
            states[i].compute_time = 0.0
            states[i].activation_size = 0.0
            states[i].parameter_size = 0.0
            for predecessor in all_predecessors:
                states[i].compute_time += ((predecessor.forward_compute_time +
                                            predecessor.backward_compute_time) / 1000.0)
                states[i].activation_size += predecessor.activation_size
                states[i].parameter_size += predecessor.parameter_size
        gr.reset()

        output_activation_sizes = [state.output_activation_size for state in states]
        all_predecessor_ids = [[states_indices[predecessor] for predecessor in
                                antichain_gr.predecessors(states[i].node_id)]
                               for i in range(len(states))]

        compute_times = []
        activation_sizes = []
        parameter_sizes = []
        for i in range(len(states)+1):
            compute_times_row = []
            activation_sizes_row = []
            parameter_sizes_row = []
            for j in range(len(states)):
                if i == 0:
                    compute_times_row.append(states[j].compute_time)
                    activation_sizes_row.append(states[j].activation_size)
                    parameter_sizes_row.append(states[j].parameter_size)
                else:
                    if j > (i-1):
                        compute_times_row.append(states[j].compute_time -
                            states[i-1].compute_time)
                        activation_sizes_row.append(states[j].activation_size -
                            states[i-1].activation_size)
                        parameter_sizes_row.append(states[j].parameter_size -
                            states[i-1].parameter_size)
                    else:
                        compute_times_row.append(None)
                        activation_sizes_row.append(None)
                        parameter_sizes_row.append(None)
            compute_times.append(compute_times_row)
            activation_sizes.append(activation_sizes_row)
            parameter_sizes.append(parameter_sizes_row)

        self.compute_times = compute_times
        self.activation_sizes = activation_sizes
        self.parameter_sizes = parameter_sizes
        self.all_predecessor_ids = all_predecessor_ids
        self.graph = gr
        self.output_activation_sizes = output_activation_sizes
        self.total_compute_time = compute_times[0][len(states)-1]
        self.states = states
        self.nodes_to_remove = nodes_to_remove

    def __str__(self):
        return ("GPU name: %s, count: %d, profile_location: %s" % (self.name, self.count, self.profile_location))

def get_servers_list_from_file(file_str):
    servers_list = []
    current_server = None
    file_str_lines = file_str.strip().split('\n')
    for file_str_line in file_str_lines:
        if file_str_line.startswith('Server'):
            server_names = file_str_line.strip().split()
            assert(len(server_names) == 1)
            if current_server != None and current_server.num_gpus != 0:
               servers_list.append(current_server)
            current_server = Server(server_names[0])
        else:
            gpu = GPU.from_str(file_str_line.strip())
            current_server.insert_gpu(gpu)
    if current_server != None and current_server.num_gpus !=0:
        servers_list.append(current_server)
    return servers_list

def get_servers_dict_from_file(file_str):
    servers_dict = {}
    current_server = None
    file_str_lines = file_str.strip().split('\n')
    for file_str_line in file_str_lines:
        if file_str_line.startswith('Server'):
            server_names = file_str_line.strip().split()
            assert(len(server_names) == 1)
            if current_server != None and current_server.num_gpus != 0:
               servers_dict[current_server.name] = current_server
            current_server = Server(server_names[0])
        else:
            gpu = GPU.from_str(file_str_line.strip())
            current_server.insert_gpu(gpu)
    if current_server != None and current_server.num_gpus !=0:
        servers_dict[current_server.name] = current_server
    return servers_dict

def create_server_from_server_list(server_list):
    up_server = Server("up_server")
    # View each server as a GPU
    for server in server_list:
        gpu = GPU(name=server.name, count=1)
        gpu.compute_times = server.compute_times
        gpu.total_compute_time = server.total_compute_time
        gpu.num_machines_within_machine = server.total_gpus
        gpu.states = list(server.dict_gpu.values())[0].states #all gpus' states are the same
        up_server.insert_gpu(gpu)
    return up_server
    


# exchange gpus if it find that there are two stages that can match
# e.g., stage1 [txp, tv]
#       stage2 [tv]
# then we can exchange them to get [tv, tv], [txp] such that it won't let one gpu idle too much
def exchange_gpu(replication_factors):
    one_gpu_stage = {}
    mul_gpu_stage = {}
    for i in range(len(replication_factors)):
        stage_info = replication_factors[i]
        if stage_info[0] == 1:
           one_gpu_stage[i] = stage_info[1]
        else:
           mul_gpu_stage[i] = stage_info[1]
    for key1 in one_gpu_stage:
        value1 = one_gpu_stage[key1]
        assert(len(value1) == 1)
        #print("value1:", value1)
        for key2 in mul_gpu_stage:
            value2 = mul_gpu_stage[key2]
            assert(len(value2) > 1)
            gpu_alone = value1[0]
            list_cp = value2
            #print("list_cp:", list_cp, gpu_alone)
            if gpu_alone not in list_cp:
                continue
            list_left = [value for value in list_cp if value != gpu_alone] #list_cp.remove(gpu_alone)
            #print("list_left:", list_left)
            if len(list_left) == 1:
               gpu1 = list_left[0]
               value1 = [gpu1 if x==gpu_alone else x for x in value1]
               value2 = [gpu_alone if x==gpu1 else x for x in value2]
               one_gpu_stage[key1] = value1
               mul_gpu_stage[key2] = value2
               replication_factors[key1] = (len(value1), value1)
               replication_factors[key2] = (len(value2), value2)
               break
    print(one_gpu_stage)
    return replication_factors


if __name__ == '__main__':
    file_name = 'hete_files/test_server.txt'
    server_dict = get_servers_dict_from_file(open(file_name, 'r').read())
    server_list = list(server_dict.values())
    for server in server_list:
        for gpu in server.dict_gpu.items():
            gpu[1].get_compute_info_from_file()
        server.create_gpu_list_based_count()
        server.create_gpu_list_based_compute()
    for server in server_list:
        server.print_server() 
    up_server = create_server_from_server_list(server_list)
    up_server.print_server()
    up_server.create_gpu_list_based_count()
    up_server.create_gpu_list_based_compute()
    print("up_server.gpu_compute_ability_list_h2l:", up_server.gpu_compute_ability_list_h2l)
    print("up_server.gpu_count_list_h2l:", up_server.gpu_count_list_h2l)
    machine_have = sum([up_server.dict_gpu[i].num_machines_within_machine for i in up_server.gpu_count_list_h2l])
    print("machine_have:", machine_have)
               
    #replication_factors = []
    #replication_factors.append((3, ['tv', 'txp', 'txp']))
    #replication_factors.append((1, ['tv']))
    #replication_factors.append((2, ['tv', 'txp']))
    #replication_factors.append((1, ['txp']))
    #replication_factors = exchange_gpu(replication_factors)
    #print(replication_factors)
                
              
