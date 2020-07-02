import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import MinkowskiEngine as ME
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

class ModelNetDataLoader(Dataset):
    def __init__(self, root,  shared_dict={}, voxel_size=0.05, split='train', uniform=False, data_augmentation=True, cache_size=15000):
        self.root = root
        self.voxel_size = voxel_size
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.data_augmentation = data_augmentation

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        shape_ids['val'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_val.txt'))]

        assert (split == 'train' or split == 'test' or split == 'val')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = shared_dict #{}  # from index to (point_set, cls) tuple
        self.data_time = AverageMeter()
        self.data_agu_time = AverageMeter()
        self.voxel_time = AverageMeter()
        #self.npoints = 4000

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
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
            if len(self.cache) < self.cache_size:
                self.cache[index] = (pts, cls)

        start = time.time()
        #choice = np.random.choice(len(pts), self.npoints, replace=True)
        #point_set = pts[choice, :]
        point_set = pts

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

        quantized_coords = np.floor(point_set / self.voxel_size)
        inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)
        feats = np.ones((len(quantized_coords[inds]), 1))
        feats = torch.from_numpy(feats)
        feats = feats.type(torch.float)
        self.voxel_time.update(time.time() - start)

        return quantized_coords[inds], feats, cls 

    def __getitem__(self, index):
        return self._get_item(index)

class Network(nn.Module):
    def __init__(self, channels, D):
        nn.Module.__init__(self)
        self.D = D
        self.conv1 = ME.MinkowskiConvolution(in_channels=channels[0],
                    out_channels=channels[1],
                    kernel_size=3,
                    stride=2,
                    dilation=1,
                    has_bias=False,
                    dimension=self.D)
        self.norm1 = ME.MinkowskiBatchNorm(channels[1])
        self.relu1 = ME.MinkowskiReLU()
        self.conv2 = ME.MinkowskiConvolution(in_channels=channels[1],
                    out_channels=channels[2],
                    kernel_size=3,
                    stride=2,
                    dilation=1,
                    has_bias=False,
                    dimension=self.D)
        self.norm2 = ME.MinkowskiBatchNorm(channels[2])
        self.relu2 = ME.MinkowskiReLU()
        self.glo_pool = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(channels[2], channels[3])
        self.weight_init()
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                nn.init.constant_(m.kernel, 0.1)
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            if isinstance(m, ME.MinkowskiLinear):
                nn.init.constant_(m.linear.weight, 0.1)
                nn.init.constant_(m.linear.bias, 0)
    def forward(self, x):
        out1 = self.relu1(self.norm1(self.conv1(x)))
        out2 = self.relu2(self.norm2(self.conv2(out1)))
        out3 = self.linear(self.glo_pool(out2))
        return out3

if __name__ == '__main__':
    import torch
    from dataset import collate_pointcloud_fn
    manager = Manager()
    shared_dict = manager.dict()
    data = ModelNetDataLoader('/extra_disk/keke/pipeDream/dataSet/modelnet40_normal_resampled', 
                              shared_dict=shared_dict, split='val', uniform=False, data_augmentation=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=64, num_workers=4, shuffle=True, collate_fn=collate_pointcloud_fn)
    criterion = nn.CrossEntropyLoss()
    channels = [1, 64, 64, 40]
    net = Network(channels, D=3)
    net = net.to('cuda')
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
    data_time = AverageMeter()
    data_transfer = AverageMeter()
    for i in range(10):
        start_epoch = time.time()
        start = time.time()
        for j, data_dict in enumerate(DataLoader):
            data_time.update(time.time() - start)
            s1 = time.time()
            coords = data_dict['coords']
            feats = data_dict['feats']
            labels = data_dict['labels']
            sin = ME.SparseTensor(
            feats, #coords[:, :3] * args.voxel_size,
            coords.int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
            )  #.to(device)
            print("coords:", coords.shape)
            sin = sin.to('cuda')
            labels = labels.to('cuda')
            torch.cuda.synchronize()
            data_transfer.update(time.time() - s1)
            print("Read data time:", i, j, data_time.val, data_time.avg, data_transfer.val, data_transfer.avg)
            sout = net(sin)
            print("sout.f:", sout.F.shape, "labels:", labels.shape)
            loss = criterion(sout.F, labels)
            loss.backward()
            optimizer.step()
            start = time.time()
        print("Epoch load time:", time.time() - start_epoch, data_time.avg)
