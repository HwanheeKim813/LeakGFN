import os
import pickle
import gzip
from torch.utils.tensorboard import SummaryWriter
import copy
import pprint
import wandb


def get_logger(args):
    if args.enable_tensorboard:
        return TensorboardLogger(args)
    elif args.enable_wandb:
        return WandbLogger(args)
    else:
        return Logger(args)

class Logger:
    def __init__(self, args):
        self.data = {}
        self.args = copy.deepcopy(vars(args))
        self.context = ""

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]

    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def update(self):
        """Update logger (no-op for base Logger, override in subclasses)"""
        pass

    def save(self, save_path, args):
        pickle.dump({'logged_data': self.data, 'args': self.args},
                    gzip.open(save_path, 'wb'))

class TensorboardLogger(Logger):
    def __init__(self, args):
        self.data = {}
        self.context = ""
        self.args = copy.deepcopy(vars(args))
        self.writer = SummaryWriter(log_dir=args.log_dir)
        # print(self.args)
        pprint.pprint(self.args)
        self.writer.add_hparams(self.args, {})

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]
        self.writer.add_scalar(key, value, len(self.data[key]))

    def add_scalars(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]
        self.writer.add_scalars(key, value, len(self.data[key]))

    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def update(self):
        """Update TensorBoard logger (flush writer)"""
        self.writer.flush()

    def save(self, save_path):
        pickle.dump({'logged_data': self.data, 'args': self.args},
                    gzip.open(save_path, 'wb'))
        self.writer.flush()

class WandbLogger(Logger):
    def __init__(self, args):
        self.data = {}
        self.context = ""
        self.args = copy.deepcopy(vars(args))
        self.current_step = 0
        # 만약 train_soo.py에서 wandb.init이 호출되지 않는 경우, 여기서 wandb.init을 호출해야 합니다.
        # 예시:
        # if not wandb.run:
        #     wandb.init(project="GFlowMoA", name=self.args.get('objectives', 'default'), config=self.args)
        pprint.pprint(self.args)
    def set_context(self, context):
        self.context = context
    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value
    def update(self):
        wandb.log(self.data)
        self.data = {}
        self.current_step += 1
    def save(self, save_path):
        pickle.dump({'logged_data': self.data, 'args': self.args},
                    gzip.open(save_path, 'wb'))
