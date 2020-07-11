# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import threading
import torch
import torch.distributed as dist

import MinkowskiEngine as ME
from typing import NamedTuple
import sys

sys.path.append("../")
import threadsafe_counter
import threadsafe_queue
import os
CPU_COUNT = os.cpu_count()
if 'OMP_NUM_THREADS' in os.environ:
    CPU_COUNT = int(os.environ['OMP_NUM_THREADS'])

NCCL='nccl'
GLOO='gloo'

class SparseDtype(NamedTuple):
    coordsDtype: str
    featsDtype: str
    strideDtype: str

def createTensorDtype(tensor):
    if isinstance(tensor, ME.SparseTensor):
        return SparseDtype(tensor.C.dtype, tensor.F.dtype, torch.int)
    else:
        return tensor.dtype

def createTensorSize(tensor):
    if isinstance(tensor, ME.SparseTensor):
        sparse_tensor_size = []
        sparse_tensor_size.append(list(tensor.C.size()))
        sparse_tensor_size.append(list(tensor.F.size()))
        sparse_tensor_size.append((len(tensor.tensor_stride),))
        return sparse_tensor_size
    else:
        return list(tensor.size())

class CommunicationSparseHandler(object):
    """ Handles communication between stages.

    For stages on different machines, use send/recv.
    For stages on same machine, use broadcast.
    """
    def __init__(self, master_addr, master_port, rank,
                 local_rank, num_ranks_in_server,
                 world_size, fp16, backend):
        """ Set up process groups.

        Note: To turn off broadcasting, set num_ranks_in_server = 1.
        """
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self.num_ranks_in_server = num_ranks_in_server
        self.world_size = world_size
        self.fp16 = fp16
        assert num_ranks_in_server > 0

        # Initialize the distributed environment.
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        assert dist.get_world_size() == self.world_size
        print("Finished initializing process group; backend: %s, rank: %d, "
              "world_size: %d" % (backend, rank, world_size))

        # Stores list of ranks of GPUs on the same server.
        self.ranks_in_server = []

        if num_ranks_in_server == 1:
            return

        # Stores information about tensors sent directly GPU-to-GPU.
        self.connection_list = []

        # Stores process groups (for broadcast() connections).
        self.process_groups = {}

        # Populate ranks_in_server.
        rank_of_first_gpu_in_server = rank - rank % num_ranks_in_server
        for connected_rank in range(
            rank_of_first_gpu_in_server,
            rank_of_first_gpu_in_server + num_ranks_in_server):
            if connected_rank == rank:
                continue
            self.ranks_in_server.append(connected_rank)
        assert len(self.ranks_in_server) == num_ranks_in_server - 1, \
            self.ranks_in_server

    def is_gpu_to_gpu_comm(self, connected_rank):
        if connected_rank in self.ranks_in_server:
            return True
        return False

    def register_tensor(self, connected_rank, tag):
        """
        Builds connections list of tensors that are communicated GPU to GPU.

        For tensors that are sent GPU-to-GPU (intra-server for GLOO backend),
        make a list of destination/source ranks and the corresponding tag.
        This information is then used to crate process groups.
        """
        if not self.is_gpu_to_gpu_comm(connected_rank=connected_rank):
            return
        connection_info = [tag, connected_rank]
        self.connection_list.append(connection_info)

    def initialize(self, receive_ranks, send_ranks,
                   tensor_tags, target_tensor_names,
                   training_tensor_dtypes,
                   rank_in_stage,
                   num_ranks_in_stage,
                   ranks_in_previous_stage,
                   ranks_in_next_stage):
        """
        Initialize state needed for CommunicationSparseHandler.
        """
        self.receive_ranks = receive_ranks
        self.send_ranks = send_ranks
        self.tensor_tags = tensor_tags
        self.target_tensor_names = target_tensor_names
        self.training_tensor_dtypes = training_tensor_dtypes
        self.rank_in_stage = rank_in_stage
        self.num_ranks_in_stage = num_ranks_in_stage
        self.ranks_in_previous_stage = ranks_in_previous_stage
        self.num_ranks_in_previous_stage = len(ranks_in_previous_stage)
        self.ranks_in_next_stage = ranks_in_next_stage
        self.num_ranks_in_next_stage = len(ranks_in_next_stage)

        self.setup_queues()
        self.setup_messaging_schedule()
        self.create_process_groups()

    def setup_queues(self):
        """
        Setup queues for communication between main compute thread
        and helper communication threads. One queue per tensor
        in forward / backward direction.
        """
        self.forward_receive_queues = {}
        self.backward_receive_queues = {}
        self.forward_send_queues = {}
        self.backward_send_queues = {}
        self.num_forward_threads = 0
        self.num_backward_threads = 0

        self.target_receive_rank_counts = {}
        self.target_send_rank_counts = {}
        # Setup queues for each tensor to be received and sent.
        for input_name in self.receive_ranks:
            self.forward_receive_queues[input_name] = []
            self.backward_send_queues[input_name] = []
            for i in range(len(self.receive_ranks[input_name])):
                self.forward_receive_queues[input_name].append(
                    threadsafe_queue.Queue())
                self.backward_send_queues[input_name].append(
                    threadsafe_queue.Queue())
                target_receive_rank = self.receive_ranks[input_name][i]
                self.register_tensor(
                    connected_rank=target_receive_rank,
                    tag=self.tensor_tags[input_name])
                if target_receive_rank not in self.target_receive_rank_counts:
                    self.target_receive_rank_counts[target_receive_rank] = 0
                self.target_receive_rank_counts[target_receive_rank] += 1
                self.num_forward_threads += 1
                self.num_backward_threads += 1
        for output_name in self.send_ranks:
            self.backward_receive_queues[output_name] = []
            self.forward_send_queues[output_name] = []
            for i in range(len(self.send_ranks[output_name])):
                self.backward_receive_queues[output_name].append(
                    threadsafe_queue.Queue())
                self.forward_send_queues[output_name].append(
                    threadsafe_queue.Queue())
                target_send_rank = self.send_ranks[output_name][i]
                self.register_tensor(
                    connected_rank=target_send_rank,
                    tag=self.tensor_tags[output_name])
                if target_send_rank not in self.target_send_rank_counts:
                    self.target_send_rank_counts[target_send_rank] = 0
                self.target_send_rank_counts[target_send_rank] += 1
                self.num_forward_threads += 1
                self.num_backward_threads += 1

        for target_tensor_name in self.target_tensor_names:
            # Queues for target in forward pass.
            self.forward_receive_queues[target_tensor_name] = []
            self.forward_send_queues[target_tensor_name] = []

            if self.num_ranks_in_previous_stage > 0:
                self.receive_ranks[target_tensor_name] = self.ranks_in_previous_stage
                for i in range(len(self.receive_ranks[target_tensor_name])):
                    self.register_tensor(
                        connected_rank=self.receive_ranks[target_tensor_name][i],
                        tag=self.tensor_tags[target_tensor_name])
                    self.forward_receive_queues[target_tensor_name].append(
                        threadsafe_queue.Queue())
                    self.num_forward_threads += 1

            if self.num_ranks_in_next_stage > 0:
                self.send_ranks[target_tensor_name] = self.ranks_in_next_stage
                for i in range(len(self.send_ranks[target_tensor_name])):
                    self.register_tensor(
                        connected_rank=self.send_ranks[target_tensor_name][i],
                        tag=self.tensor_tags[target_tensor_name])
                    self.forward_send_queues[target_tensor_name].append(
                        threadsafe_queue.Queue())
                    self.num_forward_threads += 1

        print ("Send ranks: ", self.rank, self.send_ranks)
        print ("Receive ranks: ", self.rank, self.receive_ranks)

        # Queues for ack for forward pass-only runs as a clocking mechanism.
        self.num_ack_threads = 0
        if "ack" in self.tensor_tags:
            self.backward_receive_queues["ack"] = []
            self.backward_send_queues["ack"] = []
            for i in range(self.num_ranks_in_previous_stage):
                self.register_tensor(
                    connected_rank=self.ranks_in_previous_stage[i],
                    tag=self.tensor_tags["ack"])
                self.backward_send_queues["ack"].append(
                    threadsafe_queue.Queue())
                self.num_ack_threads += 1
            for i in range(self.num_ranks_in_next_stage):
                self.register_tensor(
                    connected_rank=self.ranks_in_next_stage[i],
                    tag=self.tensor_tags["ack"])
                self.backward_receive_queues["ack"].append(
                    threadsafe_queue.Queue())
                self.num_ack_threads += 1

    def set_tensor_shapes(self, tensor_shapes):
        self.tensor_shapes = tensor_shapes

    def set_counter(self, counter):
        self.counter = threadsafe_counter.Counter(counter)

    def wait(self):
        self.counter.wait()

    def num_iterations_for_helper_threads(self, num_iterations):
        """ Scales the number of iterations a helper thread is run for.

        Since we start a helper thread for each worker in previous/next stage,
        the number of iterations for each thread should be scaled by
        the number of workers in previous/next stage.

        TODO: don't current support uneven configurations.
        """
        forward_num_iterations = num_iterations
        backward_num_iterations = num_iterations

        if self.num_ranks_in_next_stage > 0:
            assert forward_num_iterations % self.num_ranks_in_next_stage == 0
            forward_num_iterations = forward_num_iterations // \
                self.num_ranks_in_next_stage
        else:
            forward_num_iterations = 0

        if self.num_ranks_in_previous_stage > 0:
            assert backward_num_iterations % self.num_ranks_in_previous_stage == 0
            backward_num_iterations = backward_num_iterations // \
                self.num_ranks_in_previous_stage
            #print("---debug:, num_iterations_for_helper_threads, changed here")
            #backward_num_iterations = backward_num_iterations * self.num_ranks_in_stage // \
        else:
            backward_num_iterations = 0
        #print("---debug, num_iterations_for_helper_threads:", self.rank, forward_num_iterations, backward_num_iterations)

        return forward_num_iterations, backward_num_iterations

    def start_helper_threads(self, num_iterations, forward_only):
        """
        Start helper communication threads, one for each queue.
        """
        if forward_only:
            self.set_counter(self.num_forward_threads +
                             self.num_ack_threads)
            # For validation, receive acks in backward pass from next stage, send
            # acks in backward pass to next stage.
            self.receive_ranks["ack"] = self.ranks_in_previous_stage
            self.send_ranks["ack"] = self.ranks_in_next_stage
        else:
            self.set_counter(self.num_forward_threads +
                             self.num_backward_threads)
            if "ack" in self.receive_ranks:
                del self.receive_ranks["ack"]
            if "ack" in self.send_ranks:
                del self.send_ranks["ack"]

        (num_iterations_for_forward_threads,
         num_iterations_for_backward_threads) = \
            self.num_iterations_for_helper_threads(
                num_iterations=num_iterations)
        dtype = torch.float16 if self.fp16 else torch.float32

        # Setup queues for each tensor to be received and sent.
        for input_name in self.receive_ranks:
            if input_name in self.target_tensor_names or input_name == "ack":
                continue

            for i in range(len(self.receive_ranks[input_name])):
                if not forward_only:
                    self.start_helper_thread(
                        self.send_helper_thread_args,
                        send_helper_thread,
                        [input_name, i, True],
                        num_iterations_for_backward_threads)
                self.start_helper_thread(
                    self.recv_helper_thread_args,
                    recv_helper_thread,
                    [input_name,
                     i,
                     self.training_tensor_dtypes[input_name],
                     False],
                    num_iterations_for_backward_threads)
        for output_name in self.send_ranks:
            if output_name in self.target_tensor_names or output_name == "ack":
                continue

            for i in range(len(self.send_ranks[output_name])):
                if not forward_only:
                    # changed by keke here for only send gradients for SparseTensor
                    if isinstance(self.training_tensor_dtypes[output_name], SparseDtype):
                        # get the feature type
                        dtype = self.training_tensor_dtypes[output_name][1] 
                    else:
                        dtype = self.training_tensor_dtypes[output_name]
                    self.start_helper_thread(
                        self.recv_helper_thread_args,
                        recv_helper_thread,
                        [output_name, i,
                         dtype, #self.training_tensor_dtypes[output_name],
                         True],
                        num_iterations_for_forward_threads)
                self.start_helper_thread(
                    self.send_helper_thread_args,
                    send_helper_thread,
                    [output_name, i, False],
                    num_iterations_for_forward_threads)

        for target_tensor_name in self.target_tensor_names:
            if self.num_ranks_in_previous_stage > 0:
                for i in range(len(self.receive_ranks[target_tensor_name])):
                    self.start_helper_thread(
                        self.recv_helper_thread_args,
                        recv_helper_thread,
                        [target_tensor_name, i, torch.int64,
                         False],
                        num_iterations_for_backward_threads)

            if self.num_ranks_in_next_stage > 0:
                for i in range(len(self.send_ranks[target_tensor_name])):
                    self.start_helper_thread(
                        self.send_helper_thread_args,
                        send_helper_thread,
                        [target_tensor_name, i, False],
                        num_iterations_for_forward_threads)

        # Start helper threads for ack for forward pass-only run as a clocking
        # mechanism.
        if forward_only:
            if "ack" in self.receive_ranks:
                for i in range(len(self.receive_ranks["ack"])):
                    self.start_helper_thread(self.send_helper_thread_args,
                                             send_helper_thread,
                                             ["ack", i, True],
                                             num_iterations_for_backward_threads)
            if "ack" in self.send_ranks:
                for i in range(len(self.send_ranks["ack"])):
                    self.start_helper_thread(self.recv_helper_thread_args,
                                             recv_helper_thread,
                                             ["ack", i, torch.int64, True],
                                             num_iterations_for_forward_threads)

    def start_helper_thread(self, args_func, func, args_func_args, num_iterations):
        """
        Start passed-in func on a helper thread.
        """
        args_func_args += [num_iterations]
        args = args_func(*args_func_args)
        helper_thread = threading.Thread(target=func,
                                         args=args)
        helper_thread.start()

    def create_process_groups(self):
        """ Create process groups in the same order across all ranks.

        To create process groups in the same order, each worker collects
        the connection_list of all other workers. To do this, every worker
        gathers the largest size of all other worker's connection_lists (L).
        Then every worker creates a tensor of size Lx2, where each row
        represents a connection, and fills up this tensor depending on how
        large its own connection list is. The worker(s) w/ the largest
        connection list will fill up the entire tensor.

        After constructing this list, an all_gather is performed, after which
        each worker has an identical NxLx2 output, where N is the number of
        workers (world_size), and each index of output represents a worker's
        connection list. For i=self.rank, the output will be identical to the
        workers local connection list.

        Each worker then iterates in the same order over the connections list,
        checking if each connection has been created yet (every connection will
        appear twice in the output), and creating a new process group if one
        doesn't exist for that connection, for both the forward and backward
        direction. Since ranks within process groups must always be identical,
        the smaller rank always goes first, followed by the larger rank.
        """
        if self.num_ranks_in_server == 1:
            return

        print("Setting up process groups for broadcasts...")

        # Figure out the size of the largest connection list that any worker
        # has (L).
        connection_list_size = torch.tensor(
            len(self.connection_list), dtype=torch.int)
        if self.backend == NCCL:
            connection_list_size = connection_list_size.cuda()
        gathered_connection_list_sizes = [
            torch.ones_like(connection_list_size)
            for _ in range(self.world_size)]
        dist.all_gather(gathered_connection_list_sizes,
                        connection_list_size)
        max_connection_list_size = max(
            gathered_connection_list_sizes)

        if max_connection_list_size == 0:
            return 

        # Build tensor to send local connection list to all other workers.
        connection_list_tensor = torch.ones([max_connection_list_size, 2],
                                            dtype=torch.int) * -1
        if self.backend == NCCL:
            connection_list_tensor = connection_list_tensor.cuda()
        if len(self.connection_list) > 0:
            connection_list_tensor[0:len(self.connection_list)] = \
                torch.IntTensor(self.connection_list)

        # Gather connection lists of all workers.
        aggregated_connection_list = [
            torch.ones_like(connection_list_tensor)
            for _ in range(self.world_size)]
        dist.all_gather(aggregated_connection_list,
                        connection_list_tensor)

        # Construct identical process groups on each worker.
        local_rank_connections = 0
        for src_rank in range(len(aggregated_connection_list)):
            for connection in aggregated_connection_list[src_rank]:
                tag = int(connection[0])
                dst_rank = int(connection[1])

                if tag == -1:
                    assert dst_rank == -1
                    continue

                min_rank = min(src_rank, dst_rank)
                max_rank = max(src_rank, dst_rank)
                assert min_rank != max_rank

                if min_rank not in self.process_groups:
                    self.process_groups[min_rank] = {}

                if max_rank not in self.process_groups[min_rank]:
                    self.process_groups[min_rank][max_rank] = {}

                if tag not in self.process_groups[min_rank][max_rank]:
                    sub_process_group_fwd = dist.new_group(
                        ranks=[min_rank, max_rank])
                    sub_process_group_bwd = dist.new_group(
                        ranks=[min_rank, max_rank])

                    self.process_groups[min_rank][max_rank][tag] = {
                        'forward': sub_process_group_fwd,
                        'backward': sub_process_group_bwd
                    }

                    if min_rank == self.rank or max_rank == self.rank:
                        local_rank_connections += 1
        assert local_rank_connections == len(self.connection_list)

    def setup_messaging_schedule(self):
        """ Order in which to receive forward and send backwards.

        Separate indexes of ranks in previous stage based on their
        corresponding offset in this stage. Then each worker will go
        in increasing order within a subset, and process subsets in
        a decreasing order.

        This is done so that messages are processed in the order
        that they are sent. Backwards send is done so that that it
        matches up with forward receive.
        """
        self.messaging_schedule = []
        for i in range(self.num_ranks_in_stage):
            idx = i
            message_schedule = []
            # changed by keke here for unet3d 4gpu
            if self.num_ranks_in_previous_stage == 1:
                message_schedule.append(0)
            else:
                while idx < self.num_ranks_in_previous_stage:
                    message_schedule.append(idx)
                    idx += self.num_ranks_in_stage
            if len(message_schedule) > 0:
                self.messaging_schedule.append(message_schedule)

        self.fwd_messaging_scheduling_row = self.rank_in_stage
        self.fwd_messaging_scheduling_col = 0
        self.bwd_messaging_scheduling_row = self.rank_in_stage
        self.bwd_messaging_scheduling_col = 0

        # For cases where previous stage has less workers than current stage.
        while self.fwd_messaging_scheduling_row > \
            len(self.messaging_schedule):
            self.fwd_messaging_scheduling_row -= 1
            self.bwd_messaging_scheduling_row -= 1
        #print("---debug, self.messaging_schedule:", self.rank, self.messaging_schedule)

    def get_messaging_index(self, sending):
        if sending:
            connection_rank = self.messaging_schedule[
                self.bwd_messaging_scheduling_row][
                    self.bwd_messaging_scheduling_col]
        else:
            connection_rank = self.messaging_schedule[
                self.fwd_messaging_scheduling_row][
                    self.fwd_messaging_scheduling_col]
        #print("---debug, get_messaging_index:", self.rank, sending, connection_rank)

        return connection_rank

    def increment_messaging_index(self, sending):
        if sending:
            self.bwd_messaging_scheduling_col += 1
            if self.bwd_messaging_scheduling_col == len(
                    self.messaging_schedule[
                        self.bwd_messaging_scheduling_row]):
                self.bwd_messaging_scheduling_col = 0
                self.bwd_messaging_scheduling_row -= 1
                if self.bwd_messaging_scheduling_row == -1:
                    self.bwd_messaging_scheduling_row = \
                        len(self.messaging_schedule) - 1
        else:
            self.fwd_messaging_scheduling_col += 1
            if self.fwd_messaging_scheduling_col == len(
                    self.messaging_schedule[
                        self.fwd_messaging_scheduling_row]):
                self.fwd_messaging_scheduling_col = 0
                self.fwd_messaging_scheduling_row -= 1
                if self.fwd_messaging_scheduling_row == -1:
                    self.fwd_messaging_scheduling_row = \
                        len(self.messaging_schedule) - 1

    def recv_helper_thread_args(self, tensor_name, index, dtype,
                                backward, num_iterations):
        if backward:
            src_rank = self.send_ranks[tensor_name][index]
        else:
            src_rank = self.receive_ranks[tensor_name][index]

        sub_process_group = None
        tag = self.tensor_tags[tensor_name]
        if self.is_gpu_to_gpu_comm(connected_rank=src_rank) and tensor_name != "ack":
            min_rank = min(self.rank, src_rank)
            max_rank = max(self.rank, src_rank)
            if src_rank > self.rank:
                sub_process_group = \
                    self.process_groups[min_rank][max_rank][tag]['backward']
            else:
                sub_process_group = \
                    self.process_groups[min_rank][max_rank][tag]['forward']
            assert sub_process_group

        if backward:
            queue = self.backward_receive_queues[tensor_name][index]
            # changed by keke to make backward only receive SparseTensor gradient
            tensor_shape = self.tensor_shapes[tensor_name]
            if any(isinstance(el, list) for el in tensor_shape):
                tensor_shape = tensor_shape[1]
        else:
            queue = self.forward_receive_queues[tensor_name][index]
            tensor_shape = self.tensor_shapes[tensor_name]

        #print("recv_helper_thread_args:", tensor_name, backward, tensor_shape)
        #print("---debug, recv_helper_thread_args:", self.rank, backward, index, src_rank, len(queue))
        return (queue, self.counter, self.local_rank, tensor_name,
                src_rank, tag, tensor_shape, dtype, sub_process_group,
                num_iterations)

    def send_helper_thread_args(self, tensor_name, index,
                                backward, num_iterations):
        if backward:
            dst_rank = self.receive_ranks[tensor_name][index]
            num_ranks_in_connected_stage = self.num_ranks_in_previous_stage
        else:
            dst_rank = self.send_ranks[tensor_name][index]
            num_ranks_in_connected_stage = self.num_ranks_in_next_stage

        sub_process_group = None
        tag = self.tensor_tags[tensor_name]
        if self.is_gpu_to_gpu_comm(connected_rank=dst_rank) and tensor_name != "ack":
            min_rank = min(self.rank, dst_rank)
            max_rank = max(self.rank, dst_rank)
            if dst_rank > self.rank:
                sub_process_group = \
                     self.process_groups[min_rank][max_rank][tag]['forward']
            else:
                sub_process_group = \
                    self.process_groups[min_rank][max_rank][tag]['backward']
            assert sub_process_group

        if backward:
            queue = self.backward_send_queues[tensor_name][index]
        else:
            queue = self.forward_send_queues[tensor_name][index]
        #print("---debug, send_helper_thread_args:", self.rank, backward, index, dst_rank, len(queue))

        return (queue, self.counter, self.local_rank, tensor_name, self.rank,
                dst_rank, tag, sub_process_group, num_iterations)

    def recv(self, tensor_name, forward_minibatch_id,
             backward_minibatch_id, backward=False):
        if backward:
            index = (backward_minibatch_id + self.rank_in_stage) % \
                len(self.backward_receive_queues[tensor_name])
            tensor = self.backward_receive_queues[tensor_name][
                index].remove()
            #print("---debug, recv, received tensor in backward:", self.rank, tensor_name, index, len(self.backward_receive_queues[tensor_name][index]))
            return tensor
        else:
            index = self.get_messaging_index(sending=False)
            tensor = self.forward_receive_queues[tensor_name][
                index].remove()
            if tensor.dtype == torch.float32:
                #change by keke here, SparseTensor only set requires_grad to True, no return
                #tensor = tensor.requires_grad_()
                tensor.requires_grad_()
            #print("---debug, recv, received tensor in forward:", self.rank, tensor_name, index, len(self.forward_receive_queues[tensor_name][index]))
            return tensor

    def send(self, tensor_name, tensor, forward_minibatch_id,
             backward_minibatch_id, backward=False):
        if backward:
            index = self.get_messaging_index(sending=True)
            dst_rank = self.receive_ranks[tensor_name][index]
            self.backward_send_queues[tensor_name][index].add(tensor)
            #print("---debug, send, send tensor in backward:", self.rank, dst_rank, tensor_name, index, len(self.backward_send_queues[tensor_name][index]))
        else:
            index = (forward_minibatch_id + self.rank_in_stage) % \
                len(self.send_ranks[tensor_name])
            self.forward_send_queues[tensor_name][index].add(tensor)
            #print("---debug, send, send tensor in forward:", self.rank, tensor_name, index, len(self.forward_send_queues[tensor_name][index]))

def recv_helper_thread(queue, counter, local_rank, tensor_name,
                       src_rank, tag, tensor_shape, dtype,
                       sub_process_group, num_iterations):
    torch.cuda.set_device(local_rank)
    # This method is to be executed from a helper daemon thread.
    for i in range(num_iterations):
        if isinstance(dtype, SparseDtype):
            tensor = _recv_sparse(
                tensor_name, src_rank, tensor_shape=tensor_shape,
                dtype=dtype, tag=tag,
                sub_process_group=sub_process_group)
        else:
            tensor = _recv(
                tensor_name, src_rank, tensor_shape=tensor_shape,
                dtype=dtype, tag=tag,
                sub_process_group=sub_process_group)
        queue.add(tensor)
        #print("---debug, recv_helper_thread, tensor_name:", tensor_name, src_rank, num_iterations, i, len(queue))
    counter.decrement()

def send_helper_thread(queue, counter, local_rank, tensor_name,
                       src_rank, dst_rank, tag,
                       sub_process_group, num_iterations):
    torch.cuda.set_device(local_rank)
    # This method is to be executed from a helper daemon thread.
    for i in range(num_iterations):
        tensor = queue.remove()
        if isinstance(tensor, ME.SparseTensor):
            _send_sparse(tensor, tensor_name, src_rank, dst_rank,
              tag=tag,
              sub_process_group=sub_process_group)
        else:
            _send(tensor, tensor_name, src_rank, dst_rank,
              tag=tag,
              sub_process_group=sub_process_group)
        #print("---debug, send_helper_thread, tensor_name:", tensor_name, src_rank, dst_rank, num_iterations, i)
    counter.decrement()

def _recv(tensor_name, src_rank, tensor_shape=None, dtype=torch.float32,
          tensor=None, tag=None, sub_process_group=None):
    """
    Receives tensor by calling PyTorch's recv() call.

    Tensor will be copied to GPU prior to return.
    """
    assert tag is not None
    if tensor is None:
        assert tensor_shape is not None
        assert dtype is not None
        assert dtype != torch.float16

    if sub_process_group is not None:
        # Receive tensor shape.
        received_tensor_shape = torch.zeros(len(tensor_shape),
                                            dtype=torch.int)
        dist.broadcast(tensor=received_tensor_shape,
                       src=src_rank,
                       group=sub_process_group)
        received_tensor_shape = list(map(lambda x: int(x),
                                         received_tensor_shape))

        # Receive tensor.
        tensor = torch.zeros(received_tensor_shape, dtype=dtype).cuda()
        dist.broadcast(tensor=tensor,
                       src=src_rank,
                       group=sub_process_group)
    else:
        # Receive tensor shape.
        received_tensor_shape = torch.zeros(len(tensor_shape),
                                            dtype=torch.int)
        dist.recv(tensor=received_tensor_shape,
                  src=src_rank,
                  tag=tag)
        received_tensor_shape = list(map(lambda x: int(x),
                                         received_tensor_shape))

        # Receive tensor.
        tensor = torch.zeros(received_tensor_shape, dtype=dtype)
        dist.recv(tensor=tensor,
                  src=src_rank,
                  tag=tag)
        tensor = tensor.cuda()

    assert tensor.is_cuda
    return tensor

def _recv_sparse(tensor_name, src_rank, tensor_shape=None, dtype=torch.float32,
          tensor=None, tag=None, sub_process_group=None):
    """
    Receives SparseTensor by calling PyTorch's recv() call.

    Tensor will be copied to GPU prior to return.
    """
    assert tag is not None
    if tensor is None:
        assert tensor_shape is not None
        assert dtype is not None
        assert dtype != torch.float16
    assert len(tensor_shape) == 3
    coords_shape = tensor_shape[0]
    feats_shape = tensor_shape[1]
    stride_tensor_shape = tensor_shape[2]

    if sub_process_group is not None:
        # Receive coords shape and tensor
        coords_tensor_shape = torch.zeros(len(coords_shape),
                                            dtype=torch.int)
        dist.broadcast(tensor=coords_tensor_shape, src=src_rank,
                       group=sub_process_group)
        coords_tensor_shape = list(map(lambda x: int(x),
                                         coords_tensor_shape))
        coords = torch.zeros(coords_tensor_shape, dtype=dtype[0]).cuda()
        dist.broadcast(tensor=coords, src=src_rank, group=sub_process_group)
        coords = coords.cpu()
        # Receive feats shape and tensor
        feats_tensor_shape = torch.zeros(len(feats_shape),
                                            dtype=torch.int)
        dist.broadcast(tensor=feats_tensor_shape, src=src_rank,
                       group=sub_process_group)
        feats_tensor_shape = list(map(lambda x: int(x),
                                         feats_tensor_shape))
        feats = torch.zeros(feats_tensor_shape, dtype=dtype[1]).cuda()
        dist.broadcast(tensor=feats, src=src_rank, group=sub_process_group)
        # Receive tensor_stride shape and tensor
        stride_tensor_shape = torch.zeros(len(stride_tensor_shape),
                                            dtype=torch.int)
        dist.broadcast(tensor=stride_tensor_shape, src=src_rank,
                       group=sub_process_group)
        stride_tensor_shape = list(map(lambda x: int(x),
                                         stride_tensor_shape))
        stride_tensor = torch.zeros(stride_tensor_shape, dtype=dtype[2]).cuda()
        dist.broadcast(tensor=stride_tensor, src=src_rank, group=sub_process_group)
        stride_tensor = stride_tensor.cpu()

        coords_manager = ME.CoordsManager(D=coords.size(1) - 1, num_threads=CPU_COUNT)
        sparse_tensor = ME.SparseTensor(feats=feats, coords=coords, tensor_stride=stride_tensor,coords_manager=coords_manager)
    else:
        # Receive coords shape and tensor.
        coords_tensor_shape = torch.zeros(len(coords_shape),
                                            dtype=torch.int)
        dist.recv(tensor=coords_tensor_shape,
                  src=src_rank,
                  tag=tag)
        coords_tensor_shape = list(map(lambda x: int(x),
                                         coords_tensor_shape))
        coords = torch.zeros(coords_tensor_shape, dtype=dtype[0])
        dist.recv(tensor=coords,
                  src=src_rank,
                  tag=tag)

        # Receive feats shape and tensor.
        feats_tensor_shape = torch.zeros(len(feats_shape),
                                            dtype=torch.int)
        dist.recv(tensor=feats_tensor_shape,
                  src=src_rank,
                  tag=tag)
        feats_tensor_shape = list(map(lambda x: int(x),
                                         feats_tensor_shape))
        feats = torch.zeros(feats_tensor_shape, dtype=dtype[1])
        dist.recv(tensor=feats,
                  src=src_rank,
                  tag=tag)
        feats = feats.cuda()

        # Receive stride_tensor shape and tensor.
        stride_tensor_shape = torch.zeros(len(stride_tensor_shape),
                                            dtype=torch.int)
        dist.recv(tensor=stride_tensor_shape,
                  src=src_rank,
                  tag=tag)
        stride_tensor_shape = list(map(lambda x: int(x),
                                         stride_tensor_shape))
        stride_tensor = torch.zeros(stride_tensor_shape, dtype=dtype[2])
        dist.recv(tensor=stride_tensor,
                  src=src_rank,
                  tag=tag)
        coords_manager = ME.CoordsManager(D=coords.size(1) - 1, num_threads=CPU_COUNT)
        sparse_tensor = ME.SparseTensor(feats=feats, coords=coords, tensor_stride=stride_tensor, coords_manager=coords_manager)

    assert sparse_tensor.F.is_cuda
    return sparse_tensor

def _send(tensor, tensor_name, src_rank, dst_rank, tag, sub_process_group=None):
    """
    Sends tensor by calling PyTorch's send() call.

    If tensor is being sent not via broadcast(), it will
    be first copied to the CPU.
    """
    if sub_process_group is not None:
        assert tensor.is_cuda

        # Send tensor shape.
        tensor_shape = torch.tensor(tensor.shape, dtype=torch.int)
        dist.broadcast(tensor=tensor_shape, src=src_rank,
                      group=sub_process_group)

        # Send tensor.
        contiguous_tensor = tensor.detach().clone()
        dist.broadcast(tensor=contiguous_tensor.contiguous(),
                       src=src_rank,
                       group=sub_process_group)
    else:
        assert tensor.is_cuda
        tensor = tensor.cpu()

        # Send tensor shape.
        tensor_shape = torch.tensor(tensor.shape, dtype=torch.int)
        dist.send(tensor=tensor_shape, dst=dst_rank, tag=tag)

        # Send tensor.
        dist.send(tensor=tensor, dst=dst_rank, tag=tag)

# added by keke
def _send_sparse(sparse_tensor, tensor_name, src_rank, dst_rank, tag, sub_process_group=None):
    """
    Sends tensor by calling PyTorch's send() call.

    If tensor is being sent not via broadcast(), it will
    be first copied to the CPU.
    """
    coords = sparse_tensor.C
    feats = sparse_tensor.F
    tensor_stride = torch.tensor(sparse_tensor.tensor_stride, dtype=torch.int)
    
    if sub_process_group is not None:
        assert feats.is_cuda
        assert not coords.is_cuda
        assert not tensor_stride.is_cuda

        # Send coord shape.
        tensor_shape = torch.tensor(coords.shape, dtype=torch.int)
        dist.broadcast(tensor=tensor_shape, src=src_rank,
                      group=sub_process_group)
        # Send coord tensor.
        contiguous_tensor = coords.detach().clone()
        dist.broadcast(tensor=contiguous_tensor.contiguous(),
                       src=src_rank,
                       group=sub_process_group)

        # Send feats shape.
        tensor_shape = torch.tensor(feats.shape, dtype=torch.int)
        dist.broadcast(tensor=tensor_shape, src=src_rank,
                      group=sub_process_group)
        # Send feats tensor.
        contiguous_tensor = feats.detach().clone()
        dist.broadcast(tensor=contiguous_tensor.contiguous(),
                       src=src_rank,
                       group=sub_process_group)

        # Send tensor_stride shape.
        tensor_shape = torch.tensor(tensor_stride.shape, dtype=torch.int)
        dist.broadcast(tensor=tensor_shape, src=src_rank,
                      group=sub_process_group)
        # Send tensor_stride tensor.
        contiguous_tensor = tensor_stride.detach().clone()
        dist.broadcast(tensor=contiguous_tensor.contiguous(),
                       src=src_rank,
                       group=sub_process_group)
    else:
        assert feats.is_cuda
        assert not coords.is_cuda
        assert not tensor_stride.is_cuda
        feats = feats.cpu()

        # Send coords shape and tensor
        tensor_shape = torch.tensor(coords.shape, dtype=torch.int)
        dist.send(tensor=tensor_shape, dst=dst_rank, tag=tag)
        dist.send(tensor=coords, dst=dst_rank, tag=tag)
        # Send feats shape and tensor
        tensor_shape = torch.tensor(feats.shape, dtype=torch.int)
        dist.send(tensor=tensor_shape, dst=dst_rank, tag=tag)
        dist.send(tensor=feats, dst=dst_rank, tag=tag)
        # Send tensor_stride shape and tensor
        tensor_shape = torch.tensor(tensor_stride.shape, dtype=torch.int)
        dist.send(tensor=tensor_shape, dst=dst_rank, tag=tag)
        dist.send(tensor=tensor_stride, dst=dst_rank, tag=tag)
