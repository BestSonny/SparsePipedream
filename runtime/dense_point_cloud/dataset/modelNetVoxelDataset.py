import numpy as np
import os
from torch.utils.data import Dataset
import time
import torch.nn as nn
from multiprocessing import Manager
import torch

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

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

class ModelNetVoxelDataset(Dataset):
    def __init__(self,
                 root,
                 shared_dict={},
                 npoints=2500,
                 split='train',
                 voxel_size=32,
                 data_augmentation=True,
                 uniform=False):
        self.root = root
        self.cache = shared_dict
        self.npoints = npoints
        self.split = split
        self.voxel_size = voxel_size
        self.data_augmentation = data_augmentation
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        shape_ids['val'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_val.txt'))]

        assert (split == 'train' or split == 'test' or split == 'val')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], 
                         os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

    def __getitem__(self, index):
        if index in self.cache:
            pts, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            pts = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                pts = farthest_point_sample(pts, self.npoints)
            pts[:, 0:3] = pc_normalize(pts[:, 0:3])
            pts = pts[:, 0:3]
            self.cache[index] = (pts, cls)

        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)) ,0)
        point_set = point_set / dist
        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        # voxelize point_set
        # scale coordinate to [0,num_voxel)
        scale = point_set.max(axis=0) - point_set.min(axis=0)
        point_set = (point_set - point_set.min(axis=0))/(scale*1.0)
        num_voxel = self.voxel_size
        point_set = np.floor(point_set * (num_voxel - 1))
        point_set = point_set.astype(int)
        voxeled_point = [[[0 for k in range(num_voxel)] for j in range(num_voxel)] for i in range(num_voxel)]
        for i in range(point_set.shape[0]):
            m, n, f = point_set[i]
            if(voxeled_point[m][n][f] != 0):
                continue
            voxeled_point[m][n][f] = 1
        voxeled_point = torch.from_numpy(np.array(voxeled_point)).float()

        return voxeled_point.unsqueeze_(0), cls

    def __len__(self):
        return len(self.datapath)

class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.norm1 = nn.BatchNorm3d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.linear = nn.Linear(in_features=8, out_features=2, bias=True)
    def forward(self, x):
        out1 = self.relu1(self.norm1(self.conv1(x)))
        print(out1.size())
        out2 = self.avgpool(out1)
        out2 = torch.flatten(out2, 1)
        print(out2.size())
        out3 = self.linear(out2)
        return out3

if __name__ == '__main__':
    datapath = '/extra_disk/keke/pipeDream/dataSet/modelnet40_normal_resampled'

    manager = Manager()
    shared_dict = manager.dict()
    data = ModelNetVoxelDataset(root=datapath, shared_dict=shared_dict, split='val')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    net = Network()
    net = net.to('cuda')
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
    for i in range(10):
        start_epoch = time.time()
        start = time.time()
        for j, data in enumerate(DataLoader):
            input, target = data
            input, target = input.cuda(), target.cuda()
            print("Read data time:", i, j, time.time() - start)
            sout = net(input)
            start = time.time()
        print("Epoch load time:", time.time() - start_epoch)

