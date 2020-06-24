# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class RuntimeStats:
    def __init__(self, rank, forward):
        self.stats = {
            'compute_time': 0.0,
            'send_tensors': 0.0,
            'send_tensors_size': 0,
            'receive_tensors': 0.0,
            'receive_tensors_size': 0,
        }
        self.forward = forward
        self.rank = rank
        self.count = 0

    def print_stats(self):
        if self.forward:
            print_stat = "\tRank: %d, Forward Stats:" % self.rank
        else:
            print_stat = "\tRank: %d, Backward Stats:" % self.rank
        for i in sorted(self.stats):
            units = 'seconds'
            if i == 'receive_tensors_size' or i == 'send_tensors_size':
                units = 'bytes'

            mm = self.stats[i]
            if self.count == 0:
               print_stat += "self.count is 0 \t"
            else:
               mm = self.stats[i] / self.count
            print_stat += "\t %s %.3f %s" % (i, mm, units)
        print(print_stat)

    def reset_stats(self):
        for i in self.stats.keys():
            self.stats[i] = 0.0
        self.count = 0