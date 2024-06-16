import os
import os.path as osp
from PIL import Image
import six

import lmdb

import numpy as np
import torch
import pickle
from pathlib import Path


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


def read_txt(fname):
    map = {}
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    for line in content:
        img, idx = line.split(" ")
        map[img] = idx
    return map


class ImageFolderLMDB(torch.utils.data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = loads_pyarrow(txn.get(b'__len__'))
            # self.keys = umsgpack.unpackb(txn.get(b'__keys__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            print("key", self.keys[index].decode("ascii"))
            byteflow = txn.get(self.keys[index])

        unpacked = loads_pyarrow(byteflow)

        imgbuf = unpacked
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        import numpy as np
        img = Image.open(buf).convert('RGB')
        # img.save("img.jpg")
        if self.transform is not None:
            img = self.transform(img)
        im2arr = np.array(img)
        # print(im2arr.shape)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj, protocol=5)


def folder2lmdb(path_lmdb, dataset_path, write_frequency=5000):
    all_idxs = []
    root = Path(dataset_path)

    lmdb_path = osp.join(path_lmdb)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=399511627776, readonly=False,
                   meminit=False, map_async=True)

    npz_files = list(root.glob("train_data_batch_*"))
    dataset = [np.load(str(f)) for f in npz_files]
    # print(len(dataset[0]['data']))

    data = [np.transpose((d)['data'].reshape(-1, 3, 64, 64), [0, 2, 3, 1]) for d in dataset]
    # print(len(data))
    data = np.concatenate(data, axis=0)
    labels = [d['labels'] for d in dataset]
    labels = np.concatenate(labels, axis=0)
    print(len(data))
    print(len(labels))

    txn = db.begin(write=True)
    for idx, (image, label) in enumerate(zip(data, labels)):
        # print(type(data), data)
        # image, label = data[0]
        # print(image.shape)
        all_idxs.append(idx)
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow([image, label]))
        # txn.put(u'{}'.format(imgpath).encode('ascii'), dumps_pyarrow(image))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data[0])))
    print("saving")
    txn.commit()
    txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


import fire

if __name__ == '__main__':
    fire.Fire(folder2lmdb)
