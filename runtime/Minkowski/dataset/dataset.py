import os
import os.path
import open3d as o3d
import torch.utils.data as data
import torch
import numpy as np
import sys
import json
import MinkowskiEngine as ME
import time
from multiprocessing import Manager
from torchvision.transforms import Compose as VisionCompose
from scipy.linalg import expm, norm


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

class ModelNet40Dataset(data.Dataset):
    AUGMENT = None
    DATA_FILES = {
        'train': 'train_modelnet40.txt',
        'val': 'val_modelnet40.txt',
        'test': 'test_modelnet40.txt'
    }

    CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl',
        'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser',
        'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
        'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio',
        'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent',
        'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    def __init__(self, root, shared_dict={}, split='train', voxel_size=0.05, transform=None):
        self.root = root
        self.split = split
        self.files = []
        self.cache = shared_dict 
        self.data_objects = []
        self.transform = transform
        self.voxel_size = voxel_size
        self.last_cache_percent = 0
        self.files = open(os.path.join(self.root,
                                       self.DATA_FILES[split])).read().split()
        print("Loading the subset {%s} from {%s} with {%d} files" % (split, self.root, len(self.files)))
        self.density = 4000

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh_file = os.path.join(self.root, self.files[idx])
        category = self.files[idx].split('/')[0]
        label = self.CATEGORIES.index(category)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        if idx in self.cache:
            xyz = self.cache[idx]
        else:
            # Load a mesh, over sample, copy, rotate, voxelization
            assert os.path.exists(mesh_file)
            pcd = o3d.io.read_triangle_mesh(mesh_file)
            # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
            vertices = np.asarray(pcd.vertices)
            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            pcd.vertices = o3d.utility.Vector3dVector((vertices - vmin) /
                                                      (vmax - vmin).max() + 0.5)

            # Oversample points and copy
            xyz = resample_mesh(pcd, density=self.density)
            self.cache[idx] = xyz
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                #print("Cached {%s}: {%d} / {%d}: {%.2f}%" % (self.split, len(self.cache), len(self), cache_percent))
                self.last_cache_percent = cache_percent
        # Use color or other features if available
        feats = np.ones((len(xyz), 3))

        if len(xyz) < 1000:
            print("Skipping {%s}: does not have sufficient CAD sampling density after resampling: {%d}." % (mesh_file, len(xyz)))
            return None

        if self.transform:
            xyz, feats = self.transform(xyz, feats)

        # Get coords
        coords = np.floor(xyz / self.voxel_size)
        inds = ME.utils.sparse_quantize(coords, return_index=True)
        coords = torch.from_numpy(coords)
        coords = coords.type(torch.int) 
        feats = torch.from_numpy(xyz) 
        feats = feats.type(torch.float) 

        return coords[inds], feats[inds], label

def resample_mesh(mesh_cad, density=1):
    '''
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.
    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud
    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    '''
    faces = np.array(mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                         vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross**2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)
    # face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Bug fix by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc:acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (1 - np.sqrt(r[:, 0:1])) * A + \
        np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + \
        np.sqrt(r[:, 0:1]) * r[:, 1:] * C

    return P

class Compose(VisionCompose):
    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

class RandomRotation:
    def __init__(self, axis=None, max_theta=180):
        self.axis = axis
        self.max_theta = max_theta

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, coords, feats):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180) * 2 *
                    (np.random.rand(1) - 0.5))
        R_n = self._M(
            np.random.rand(3) - 0.5,
            (np.pi * 15 / 180) * 2 * (np.random.rand(1) - 0.5))
        return coords @ R @ R_n, feats

class RandomScale:
    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords, feats):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s, feats

class RandomShear:
    def __call__(self, coords, feats):
        T = np.eye(3) + 0.1 * np.random.randn(3, 3)
        return coords @ T, feats

class RandomTranslation:
    def __call__(self, coords, feats):
        trans = 0.05 * np.random.randn(1, 3)
        return coords + trans, feats

def make_data_loader_modelnet40(root, shared_dict, split, augment_data, batch_size, shuffle, num_workers,
                                repeat, voxel_size):
    transformations = []
    if augment_data:
        transformations.append(RandomRotation(axis=np.array([0, 0, 1])))
        transformations.append(RandomTranslation())
        transformations.append(RandomScale(0.8, 1.2))
        transformations.append(RandomShear())

    dset = ModelNet40Dataset(root, shared_dict,
        split, transform=Compose(transformations), voxel_size=voxel_size)

    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_pointcloud_fn,
        'pin_memory': False,
        'drop_last': False
    }

    if repeat:
        args['sampler'] = InfSampler(dset, shuffle)
    else:
        args['shuffle'] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)
    return loader


if __name__ == '__main__':
    manager = Manager()
    shared_dict = manager.dict()

    train_transpose = Compose([RandomRotation(axis=np.array([0, 0, 1])),
                               RandomTranslation(),
                               RandomScale(0.8, 1.2), 
                               RandomShear()])
    train_dataset = ModelNet40Dataset(root="/extra_disk/keke/pipeDream/dataSet/ModelNet40",
                                      shared_dict=shared_dict,
                                      split='train',
                                      voxel_size=0.05,
                                      transform=train_transpose)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=32,
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
