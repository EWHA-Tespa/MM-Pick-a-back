import sys
import torch
import logging


class Optimizers(object):
    def __init__(self):
        self.optimizers = []
        self.lrs = []

    def add(self, optimizer, lr):
        self.optimizers.append(optimizer)
        self.lrs.append(lr)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def __getitem__(self, index):
        return self.optimizers[index]

    def __setitem__(self, index, value):
        self.optimizers[index] = value


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.).cuda()
        self.n = torch.tensor(0.).cuda()
        return

    def update(self, val, num):
        self.sum += val * num
        self.n += num

    @property
    def avg(self):
        return self.sum / self.n


def classification_accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def set_dataset_paths(train_path, val_path, dataset):

    if not train_path:
        train_path = '/data_library/%s/image/train' % (dataset)
        # train_path = '/data_library/%s/caption/train' % (dataset)

    if not val_path:
        if (dataset in ['imagenet', 'face_verification', 'emotion', 'gender'] or
            dataset[:3] == 'age'):

            val_path = '/data_library/%s/image/val' % (dataset)
            # val_path = '/data_library/%s/caption/val' % (dataset)
        else:
            val_path = '/data_library/%s/image/test' % (dataset)
            # val_path = '/data_library/%s/caption/test' % (dataset)


def set_dataset_paths_1param(args):

    if not args.train_path:
        args.train_path = '/data_library/n24news/image/train/%s' % (args.dataset)
        # args.train_path = '/data_library/n24news/caption/train/%s' % (args.dataset)

    if not args.val_path:
        if (args.dataset in ['imagenet', 'face_verification', 'emotion', 'gender'] or
            args.dataset[:3] == 'age'):
            args.val_path = '/data_library/n24news/image/val/%s' % (args.dataset)
            # args.val_path = '/data_library/n24news/caption/val/%s' % (args.dataset)
        else:
            args.val_path = '/data_library/n24news/image/test/%s' % (args.dataset)
            # args.val_path = '/data_library/n24news/caption/test/%s' % (args.dataset)


def set_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return
