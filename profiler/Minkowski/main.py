import sys; sys.path = [".."] + sys.path
import torchmodules.torchgraph as torchgraph
import torchmodules.torchlogger as torchlogger
import torchmodules.torchprofiler as torchprofiler
import torchmodules.torchsummary as torchsummary
from multiprocessing import Manager
from dataset.dataset import ModelNetMinkowski

import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import logging
import sys
import json
import argparse
import models.vgg_mink as vgg
import torch.optim as optim
import time
import MinkowskiEngine as ME
import random
from collections import OrderedDict

model_names = ['minkunet34c', 'minkunet14d', 'minkvgg']

parser = argparse.ArgumentParser(description='PyTorch Sparse Point Cloud Training')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_ngpu', type=int, default=2)
parser.add_argument('--voxel_size', type=float, default=0.02)
parser.add_argument('--max_iter', type=int, default=120000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--stat_freq', type=int, default=10)
parser.add_argument('--load_optimizer', type=str, default='true')
parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--arch', '-a', metavar='ARCH', default='minkvgg',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: minkunet34c)')
parser.add_argument('-v', '--verbose', action='store_true',
                    help="Controls verbosity while profiling")
parser.add_argument('--profile_directory', default="profiles/",
                            help="Profile directory")

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])

def create_graph(model, train_loader, summary, directory):
    """Given a model, creates and visualizes the computation DAG
       of the model in the passed-in directory."""
    graph_creator = torchgraph.GraphCreatorForMink(model, summary, module_whitelist=['MinkowskiBatchNorm', 'MinkowskiReLU', 'MinkowskiLinear', 'Dropout'])
    graph_creator.hook_modules(model)
    for i, data_dict in enumerate(train_loader):
        coords = data_dict['coords']
        feats = data_dict['feats']
        labels = data_dict['labels']
        sin = ME.SparseTensor(
            feats,
            coords.int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        )  #.to(device)
        sin = sin.to("cuda")
        output = model(sin)
        if i >= 0:
            break
    graph_creator.unhook_modules()
    graph_creator.persist_graph(directory)

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 shared_dict={},
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True,
                 voxel_size = 0.05):
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        self.voxel_size = voxel_size
        self.cache = shared_dict

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        # from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid +'.pts'),
                                                         os.path.join(self.root, category, 'points_label', uuid +'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print("classes:", self.classes)
        with open('/extra_disk/keke/pipeDream/pipedream/runtime/point_cloud/utils/num_seg_classes.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print("seg_classes:", self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        if index in self.cache:
            if self.classification:
                point_set, cls = self.cache[index]
            else:
                point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            if self.classification:
                cls = self.classes[self.datapath[index][0]]
                point_set = np.loadtxt(fn[1]).astype(np.float32)
                self.cache[index] = (point_set, cls)
            else:
                cls = self.classes[self.datapath[index][0]]
                point_set = np.loadtxt(fn[1]).astype(np.float32)
                seg = np.loadtxt(fn[2]).astype(np.int64)
                self.cache[index] = (point_set, cls, seg)
        #print(point_set.shape, seg.shape)
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)) ,0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)] ,[np.sin(theta), np.cos(theta)]])
            point_set[: ,[0 ,2]] = point_set[: ,[0 ,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        #seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))


        quantized_coords = np.floor(point_set / self.voxel_size)
        inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)
        #print("quantized_coords:", quantized_coords)

        if self.classification:
            return quantized_coords[inds], quantized_coords[inds], cls
        else:
            seg = torch.from_numpy(seg)
            return quantized_coords[inds], quantized_coords[inds], seg[inds]

    def __len__(self):
        return len(self.datapath)

def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    coords, feats, labels = list(zip(*list_data))

    eff_num_batch = len(coords)
    assert len(labels) == eff_num_batch

    coords_batch, feats_batch, quantized_labels = ME.utils.sparse_collate(coords, feats, labels)

    # Concatenate all lists
    return {
        'coords': coords_batch,
        'feats': feats_batch,
        'labels': quantized_labels,
    }

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        if val < 0:
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    global args
    args = parser.parse_args()
    args.profile = True
    #model = munet.MinkUNet34C(3, 16, D=3) #MinkUNet34C(3, 16, D=3)
    model = vgg.vgg16_bn(in_channels=1, out_channels=40, D=3) 
    print("Profiling model mink-vgg16_bn, voxel size:", args.voxel_size, " batch size", args.batch_size)
    model.cuda()

    manager = Manager()
    shared_dict = manager.dict()
    dataset = ModelNetMinkowski(basedir=args.dataset,
                                split='train',
                                voxel_size=args.voxel_size)

    train_dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=int(args.num_workers),
                                        collate_fn=collate_pointcloud_fn)
    optimizer = optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    criterion = torch.nn.CrossEntropyLoss()

    for i, data_dict in enumerate(train_dataloader):
        coords = data_dict['coords']
        feats = data_dict['feats']
        labels = data_dict['labels']
        sin = ME.SparseTensor(
                feats,
                coords.int(),
                allow_duplicate_coords=True,  # for classification, it doesn't matter
            )
        
        model_input = sin.to("cuda")
        if i >= 0:
            break
    summary = torchsummary.summary(model=model,
                module_whitelist=['MinkowskiBatchNorm', 'MinkowskiReLU', 'MinkowskiLinear', 'Dropout'],
                #module_whitelist=[],
                model_input=(model_input,),
                verbose=args.verbose, device="cuda")
    per_layer_times, data_time = profile_train(train_dataloader, model, criterion, optimizer)
    summary_i = 0
    per_layer_times_i = 0
    while summary_i < len(summary) and per_layer_times_i < len(per_layer_times):
        summary_elem = summary[summary_i]
        per_layer_time = per_layer_times[per_layer_times_i]
        if str(summary_elem['layer_name']) != str(per_layer_time[0]):
            summary_elem['forward_time'] = 0.0
            summary_elem['backward_time'] = 0.0
            summary_i += 1
            continue
        summary_elem['forward_time'] = per_layer_time[1]
        summary_elem['backward_time'] = per_layer_time[2]
        summary_i += 1
        per_layer_times_i += 1
    summary.append(OrderedDict())
    summary[-1]['layer_name'] = 'Input0'
    summary[-1]['forward_time'] = data_time
    summary[-1]['backward_time'] = 0.0
    summary[-1]['nb_params'] = 0.0
    summary[-1]['output_shape'] = [args.batch_size] + list(model_input.size()[1:])
    create_graph(model, train_dataloader, summary,
                 os.path.join(args.profile_directory, args.arch))
    print("...done!")
    return
 
def profile_train(train_loader, model, criterion, optimizer):
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    compute_time_meter = AverageMeter()
    backward_time_meter = AverageMeter()
    loss_time_meter = AverageMeter()
    NUM_STEPS_TO_PROFILE = 100  # profile 100 steps or minibatches
    NUM_STEPS_TO_WARMUP = 50  # profile 100 steps or minibatches

    model.train()
    print("model:", model)

    layer_timestamps = []
    data_times = []

    iteration_timestamps = []
    opt_step_timestamps = []
    data_timestamps = []

    start_time = time.time()
    # first loop to load data to cache
    mm = time.time()
    for i, data_dict in enumerate(train_loader):
        coords = data_dict['coords']
        feats = data_dict['feats']
        labels = data_dict['labels']
        sin = ME.SparseTensor(
            feats, #coords[:, :3] * args.voxel_size,
            coords.int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
            )  #.to(device)
        sin = sin.to('cuda')
        labels = labels.to('cuda')
        torch.cuda.synchronize()
        #print(" batch :", i, "time:", time.time() - mm)
        mm = time.time()
    print("Loading data time:", time.time()-start_time)

    for i, data_dict in enumerate(train_loader):
        coords = data_dict['coords']
        feats = data_dict['feats']
        labels = data_dict['labels']
        sin = ME.SparseTensor(
            feats,
            coords.int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        )  #.to(device)
        sin = sin.to("cuda")
        labels = labels.to("cuda")
        sout = model(sin)
        loss = criterion(sout.F, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i >= NUM_STEPS_TO_WARMUP:
            break
    
    start_time = time.time()
    for i, data_dict in enumerate(train_loader):
        coords = data_dict['coords']
        feats = data_dict['feats']
        labels = data_dict['labels']
        data_pid = os.getpid()
        data_time = time.time() - start_time
        data_time_meter.update(data_time)
        with torchprofiler.Profiling(model, module_whitelist=['MinkowskiConvolution', 'MinkowskiBatchNorm', 'MinkowskiReLU', 'MinkowskiLinear', 'Dropout']) as p:
            aa_time = time.time()
            sin = ME.SparseTensor(
                feats,
                coords.int(),
                allow_duplicate_coords=True,  # for classification, it doesn't matter
            )  #.to(device)
            sin = sin.to("cuda")
            labels = labels.to("cuda")
            data_transfer_time = time.time() - aa_time
            data_time = data_time + data_transfer_time
            if args.verbose:
                print("data_transfer_time:", i, data_transfer_time)

            st3 = time.time()
            sout = model(sin)
            compute_time = time.time() - st3
            compute_time_meter.update(compute_time)
 
            st3 = time.time()
            loss = criterion(sout.F, labels)
            loss_time = time.time() - st3
            loss_time_meter.update(loss_time)

            # compute gradient and do SGD step
            st3 = time.time()
            optimizer.zero_grad()
            loss.backward()
            if args.verbose:
                print("backward_time:", i, time.time() - st3)
            backward_time_meter.update(time.time() - st3)

            optimizer_step_start = time.time()
            optimizer.step()

            end_time = time.time()
            iteration_time = end_time - start_time
            batch_time_meter.update(iteration_time)

            if i >= NUM_STEPS_TO_PROFILE:
                break
        p_str = str(p)
        layer_timestamps.append(p.processed_times())
        data_times.append(data_time)

        if args.verbose:
            print('End-to-end time: total time: {batch_time.val:.3f} s ({batch_time.avg:.3f}) s, forward compute time: {compute_time.val:.3f} s ({compute_time.avg:.3f}) s, backward time: {backward_time.val: .3f} s ({backward_time.avg:.3f}) s, loss time: {loss_time.val:.3f} s ({loss_time.avg:.3f} s), data time: {data_time.val:.3f}s ({data_time.avg:.3f} s)'
                .format(batch_time=batch_time_meter, compute_time=compute_time_meter, backward_time=backward_time_meter, loss_time=loss_time_meter, data_time=data_time_meter))

        iteration_timestamps.append({"start": start_time * 1000 * 1000,
                                     "duration": iteration_time * 1000 * 1000})
        opt_step_timestamps.append({"start": optimizer_step_start * 1000 * 1000,
                                    "duration": (end_time - optimizer_step_start) * 1000 * 1000, "pid": os.getpid()})
        data_timestamps.append({"start":  start_time * 1000 * 1000,
                                "duration": data_time * 1000 * 1000, "pid": data_pid})
        
        start_time = time.time()
        layer_times = []
    tot_accounted_time = 0.0
    if args.verbose:
        print("\n==========================================================")
        print("Layer Type    Forward Time (ms)    Backward Time (ms)")
        print("==========================================================")

    for i in range(len(layer_timestamps[0])):
        layer_type = str(layer_timestamps[0][i][0])
        layer_forward_time_sum = 0.0
        layer_backward_time_sum = 0.0
        for j in range(len(layer_timestamps)):
            layer_forward_time_sum += (layer_timestamps[j][i][2] / 1000)
            layer_backward_time_sum += (layer_timestamps[j][i][5] / 1000)
        layer_times.append((layer_type, layer_forward_time_sum / len(layer_timestamps),
                                    layer_backward_time_sum / len(layer_timestamps)))
        if args.verbose:
            print(layer_times[-1][0], layer_times[-1][1], layer_times[-1][2])
        tot_accounted_time += (layer_times[-1][1] + layer_times[-1][2])

    print()
    print("Total accounted time: %.3f ms, data_times: %.3f ms" % (tot_accounted_time, (sum(data_times) * 1000.0) / len(data_times)))
    return layer_times, (sum(data_times) * 1000.0) / len(data_times)




if __name__ == '__main__':
    main()

    #args = parser.parse_args()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = MinkUNet34C(3, 16, D=3)
    #model.to(device)
    #train(model, device, args)
