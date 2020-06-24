import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
import json
import MinkowskiEngine as ME
import time
from multiprocessing import Manager


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
        self.data_time = AverageMeter()
        self.data_agu_time = AverageMeter()
        self.voxel_time = AverageMeter()
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
                self.meta[self.id2cat[category]].append(
                    (os.path.join(self.root, category, 'points', uuid +'.pts'),
                     os.path.join(self.root, category, 'points_label', uuid +'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print("classes:", self.classes)
        with open('misc/num_seg_classes.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print("seg_classes:", self.seg_classes, self.num_seg_classes)
        
    def __getitem__(self, index):
        start = time.time()
        if index in self.cache:
            point_set, cls = self.cache[index]
            #print("Cached index:", index)
        else:
            fn = self.datapath[index] 
            cls = self.classes[self.datapath[index][0]]
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            self.cache[index] = (point_set, cls)
            #print("New index:", index, "cache length:", len(self.cache))
        self.data_time.update(time.time() - start)

        start = time.time()
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)) ,0)
        point_set = point_set / dist  # scale
 
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)] ,[np.sin(theta), np.cos(theta)]])
            point_set[: ,[0 ,2]] = point_set[: ,[0 ,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        point_set = torch.from_numpy(point_set) 
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        self.data_agu_time.update(time.time() - start)

        start = time.time()
        quantized_coords = np.floor(point_set / self.voxel_size)
        inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)
        self.voxel_time.update(time.time() - start)
        #print("index:", index, self.data_time.avg, self.data_agu_time.avg, self.voxel_time.avg)

        if self.classification:
            return quantized_coords[inds], quantized_coords[inds], cls
        else:
            seg = np.loadtxt(fn[2]).astype(np.int64)
            seg = torch.from_numpy(seg)
            return quantized_coords[inds], quantized_coords[inds], seg[inds]

    def __len__(self):
        return len(self.datapath)

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
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def collate_pointcloud_fn(list_data):
    start = time.time()
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
    #print("collate_pointcloud_fn time:", time.time() - start)

    # Concatenate all lists
    return {
        'coords': coords_batch,
        'feats': feats_batch,
        'labels': quantized_labels,
    }

if __name__ == '__main__':
    manager = Manager()
    shared_dict = manager.dict()
    train_dataset = ShapeNetDataset(root="/extra_disk/keke/Minkowski/Dataset/shapenetcore_partanno_segmentation_benchmark_v0",
                                    shared_dict=shared_dict,
                                    classification=True,
                                    split='train',
                                    voxel_size=0.05)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=2,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True,
                                        collate_fn=collate_pointcloud_fn)
    print("len(train_dataset):", len(train_dataset))
    print("len(train_loader)", len(train_loader))
    data_time = AverageMeter()
    for i in range(10):
        start_epoch = time.time()
        start = time.time()
        for j, data_dict in enumerate(train_loader):
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
            data_time.update(time.time() - start)
            #print("Read data time:", i, j, data_time.val, data_time.avg)
            start = time.time()
        print("Epoch load time:", time.time() - start_epoch, data_time.avg)
