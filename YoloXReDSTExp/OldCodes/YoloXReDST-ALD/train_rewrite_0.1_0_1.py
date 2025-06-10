# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 17:00:15 2022

@author: Modified by Cheng Yuxuan
"""
import os
# import time
import argparse
import random
import warnings
# import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import sys
sys.path.append(r'/mnt/yoloxredstald')

from loguru import logger
from datetime import timedelta

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.exp.build_rewrite import Exp
import yolox.utils.dist as comm
from yolox.utils import (MeterBuffer,     # yolox/utils/metric.py # 不加这个括号就会报错IndentationError: unexpected indent
                         occupy_mem,      # yolox/utils/metric.py
                         gpu_mem_usage,   # yolox/utils/metric.py
                         ModelEMA,        # yolox/utils/ema.py
                         is_parallel,     # yolox/utils/ema.py
                         WandbLogger,     # yolox/utils/logger.py
                         setup_logger,    # yolox/utils/logger.py
                         all_reduce_norm, # yolox/utils/all_reduce_norm.py
                         # get_model_info,  # yolox/utils/model_utils.py
                         get_local_rank,  # yolox/utils/dist.py
                         get_world_size,  # yolox/utils/dist.py
                         get_num_devices, # yolox/utils/dist.py
                         synchronize,     # yolox/utils/dist.py
                         get_rank,        # yolox/utils/dist.py
                         load_ckpt,       # yolox/utils/checkpoint.py
                         save_checkpoint, # yolox/utils/checkpoint.py
                         configure_nccl,  # yolox/utils/setup_env.py
                         configure_omp)   # yolox/utils/setup_env.py


DEFAULT_TIMEOUT = timedelta(minutes=30)

"""Cover tools/train.py, yolox/core/launch.py, yolox/core/trainer.py, yolox/data/data_prefetcher.py"""


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(self.file_name,distributed_rank=self.rank,filename="train_log.txt",mode="a")
        
    def train(self):
        # before_train
        # logger.info("args: {}".format(self.args)) # 只恨没早点把这三个玩意注释掉
        # logger.info("exp value:\n{}".format(self.exp))
        
        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        # logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        
        self.train_loader = self.exp.get_data_loader(batch_size=self.args.batch_size,
                                                     is_distributed=self.is_distributed,
                                                     no_aug=self.no_aug,
                                                     cache_img=self.args.cache)
        self.train_loader_ir = self.exp.get_data_loader_ir(batch_size=self.args.batch_size * 4,
                                                        is_distributed=self.is_distributed,
                                                        no_aug=self.no_aug,
                                                        cache_img=self.args.cache)
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher_regular = DataPrefetcher(self.train_loader)
        self.prefetcher_irregular = DataPrefetcher(self.train_loader_ir)
        
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter)
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(batch_size=self.args.batch_size, is_distributed=self.is_distributed)
        
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                wandb_params = dict()
                for k, v in zip(self.args.opts[0::2], self.args.opts[1::2]):
                    if k.startswith("wandb-"):
                        wandb_params.update({k.lstrip("wandb-"): v})
                self.wandb_logger = WandbLogger(config=vars(self.exp), **wandb_params)
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        # logger.info("\n{}".format(model))
        
        # 初始化损失值累加器和布尔变量, batch_counter正常批次计数器, iter_unnorm非正常批次总数
        self.loss_sum_small, self.loss_sum_medium, self.loss_sum_large = 0.1, 0.1, 0.1 # loss_log = []
        self.batch_flag, self.is_regular, self.batch_counter, self.iter_normal, self.iter_unnorm = [0], True, 0, 0, 0
        # zero_values, first_values, second_values = 0, 0, 0
        
        # train_in_iter 外层for循环
        for self.epoch in range(self.start_epoch, self.max_epoch):
            # before_epoch
            logger.info("---> start train epoch{}".format(self.epoch + 1))
            
            self.prefetcher = self.prefetcher_regular
            
            if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
                logger.info("--->No mosaic aug now!")
                self.train_loader.close_mosaic()
                logger.info("--->Add additional L1 loss now!")
                if self.is_distributed:
                    self.model.module.head.use_l1 = True
                else:
                    self.model.head.use_l1 = True
                self.exp.eval_interval = 1
                if not self.no_aug:
                    self.save_ckpt(ckpt_name="last_mosaic_epoch")
            
            # train_in_iter 内层for循环
            for self.iter in range(self.max_iter):
                # 首先检查是否在正常批次内
                # if self.is_regular == True:
                    # if self.batch_counter % 10 == 0: # 每10个正常batch检查一次损失值之间的差异
                        # self.batch_flag = self.get_aug_scale()
                        # zero_values, first_values, second_values = self.get_loss_values()
                        # if int(first_values) * 0.5 <= int(zero_values) or int(second_values) * 0.5 <= int(zero_values):
                    # if self.loss_sum_small / (self.loss_sum_small + self.loss_sum_medium + self.loss_sum_large) < 0.1:
                        # self.is_regular, self.batch_counter = False, 0
                    
                    # if self.batch_flag[0] != 0:
                        # self.is_regular, self.batch_counter, temp_iter = False, 0, self.iter 
                # else:
                    # zero_values, first_values, second_values = self.get_loss_values() # 检查损失值差距是否缩小至负数
                    # if int(first_values) * 0.5 <= int(zero_values) or int(second_values) * 0.5 <= int(zero_values) or self.iter - temp_iter == 5:
                    # if self.loss_sum_small / (self.loss_sum_small + self.loss_sum_medium + self.loss_sum_large) >= 0.1:
                        # self.batch_flag, self.is_regular = [0], True
                
                if self.loss_sum_small / (self.loss_sum_small + self.loss_sum_medium + self.loss_sum_large) < 0.1:
                    self.batch_flag = [1]
                else:
                    self.batch_flag = [0]
                
                # 判断进入哪种模式
                if self.batch_flag[0] == 0:                     # train_one_iter regular
                    self.prefetcher = self.prefetcher_regular
                    # self.batch_counter = self.batch_counter + 1 # 正常batch的计数器加1,用于每三个轮次判断一次损失值差异
                    self.iter_normal = self.iter_normal + 1     # 正常batch计数器加1,用于统计总正常batch数量
                else:
                    self.prefetcher = self.prefetcher_irregular
                    self.iter_unnorm = self.iter_unnorm + 1
                
                # elif self.batch_flag[0] == 'small':             # train_one_iter small
                    # self.prefetcher = self.prefetcher_irregular
                    # self.iter_unnorm = self.iter_unnorm + 1
                # elif self.batch_flag[0] == 'medium':
                    # self.prefetcher = self.prefetcher_regular
                    # self.iter_normal = self.iter_normal + 1
                # else:                                           # train_one_iter large
                    # self.prefetcher = self.prefetcher_regular
                    # self.iter_normal = self.iter_normal + 1
                
                # iter_start_time = time.time()
                
                inps, targets = self.prefetcher.next() # inps&targets均为张量,[2,3,640,640]&[2,120,5]
                inps = inps.to(self.data_type)
                targets = targets.to(self.data_type)
                targets.requires_grad = False
                inps, targets = self.exp.preprocess(inps, targets, self.input_size)
                
                # data_end_time = time.time()
                
                with torch.cuda.amp.autocast(enabled=self.amp_training):
                    outputs = self.model(inps, targets)
                
                loss = outputs["total_loss"]
                self.loss_sum_small = self.loss_sum_small + float(outputs["loss_scale"][0])
                self.loss_sum_medium = self.loss_sum_medium + float(outputs["loss_scale"][1])
                self.loss_sum_large = self.loss_sum_large + float(outputs["loss_scale"][2])
                del outputs["loss_scale"]
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.use_model_ema:
                    self.ema_model.update(self.model)

                lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                # iter_end_time = time.time()
                
                # self.meter.update(iter_time=iter_end_time - iter_start_time, data_time=data_end_time - iter_start_time, lr=lr, **outputs)
                self.meter.update(lr=lr, **outputs)
                
                # after_iter log needed information
                if (self.iter + 1) % self.exp.print_interval == 0:
                    # TODO check ETA logic
                    # left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
                    # eta_seconds = self.meter["iter_time"].global_avg * left_iters
                    # eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

                    progress_str = "epoch: {}/{}, iter: {}/{}".format(self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter)
                    loss_meter = self.meter.get_filtered_meter("loss")
                    loss_str = ", ".join(["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()])

                    # time_meter = self.meter.get_filtered_meter("time")
                    # time_str = ", ".join(["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()])

                    # logger.info("{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(progress_str, gpu_mem_usage(), time_str, loss_str,
                                # self.meter["lr"].latest) + (", size: {:d}, {}".format(self.input_size[0], eta_str)))
                                
                    logger.info("{}, mem: {:.0f}Mb, {}, lr: {:.3e}".format(progress_str, gpu_mem_usage(), loss_str, self.meter["lr"].latest) + \
                                (", re_batch: {}, ir_batch: {}, s_loss: {}, m_loss: {}, l_loss: {}".format(self.iter_normal, self.iter_unnorm, 
                                self.loss_sum_small, self.loss_sum_medium, self.loss_sum_large)))

                    if self.rank == 0:
                        if self.args.logger == "wandb":
                            self.wandb_logger.log_metrics({k: v.latest for k, v in loss_meter.items()})
                            self.wandb_logger.log_metrics({"lr": self.meter["lr"].latest})

                    self.meter.clear_meters()
                
                # random resizing
                # if (self.progress_in_iter + 1) % 10 == 0:
                    # self.input_size = self.exp.random_resize(self.train_loader, self.epoch, self.rank, self.is_distributed)
            
            # after_epoch 内层for循环结束
            self.save_ckpt(ckpt_name="latest")
            
            if (self.epoch + 1) % self.exp.eval_interval == 0:
                all_reduce_norm(self.model)
                self.evaluate_and_save_model()
              
                
        # after_train 外层for循环结束
        logger.info("Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100))
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()
                
    def get_aug_scale(self):
        self.loss_log = {'small': self.loss_sum_small, 'medium': self.loss_sum_medium, 'large': self.loss_sum_large}
        
        return sorted(self.loss_log, key=self.loss_log.get)
    
    def get_loss_values(self):
        if self.batch_flag[0] == 'small':
            a, b, c = self.loss_sum_small, self.loss_sum_medium, self.loss_sum_large
        elif self.batch_flag[0] == 'medium':
            a, b, c = self.loss_sum_medium, self.loss_sum_small, self.loss_sum_large
        elif self.batch_flag[0] == 'large':
            a, b, c = self.loss_sum_large, self.loss_sum_medium, self.loss_sum_small

        return a, b, c
    
    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt
            ckpt = torch.load(ckpt_file, map_location=self.device)
            
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            
            # resume the training states variables
            start_epoch = (self.args.start_epoch - 1 if self.args.start_epoch is not None else ckpt["start_epoch"])
            self.start_epoch = start_epoch
            logger.info("loaded checkpoint '{}' (epoch {})".format(self.args.resume, self.start_epoch))  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        # ap50_95, ap50, summary = self.exp.eval(evalmodel, self.evaluator, self.is_distributed)
        ap50_95, ap50, summary = self.evaluator.evaluate(evalmodel, self.is_distributed, half=False)
        update_best_ckpt = ap50 > self.best_ap
        self.best_ap = max(self.best_ap, ap50)

        self.model.train()
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({"val/COCOAP50": ap50,
                                               "val/COCOAP50_95": ap50_95,
                                               "epoch": self.epoch + 1})
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {"start_epoch": self.epoch + 1,
                          "model": save_model.state_dict(),
                          "optimizer": self.optimizer.state_dict(),
                          "best_ap": self.best_ap}
            save_checkpoint(ckpt_state,update_best_ckpt,self.file_name,ckpt_name)

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(self.file_name, ckpt_name, update_best_ckpt)


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader): # 输入的loader是yolox_base.py的get_data_loader函数
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url",default=None,type=str,help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("-f","--exp_file",default=None,type=str,help="plz input your experiment description file")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-e","--start_epoch",default=None,type=int,help="resume training start epoch")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("--fp16",dest="fp16",default=False,action="store_true",help="Adopting mix precision training.")
    parser.add_argument("--cache",dest="cache",default=False,action="store_true",help="Caching imgs to RAM for fast training.")
    parser.add_argument("-o","--occupy",dest="occupy",default=False,action="store_true",help="occupy GPU memory first for training.")
    parser.add_argument("-l","--logger",type=str,help="Logger to be used for metrics",default="tensorboard")
    parser.add_argument("opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER)
    return parser


@logger.catch
def main(exp, args):
    # 设定每次使用相同的随机数种子使得CNN每次初始化一致,每次训练结果一致
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed) # 为CPU设置固定的随机数种子
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably! You may see unexpected behavior "
                      "when restarting from checkpoints.")

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()


def _distributed_worker(local_rank,main_func,world_size,num_gpus_per_machine,machine_rank,backend,dist_url,args,timeout=DEFAULT_TIMEOUT,):
    assert (torch.cuda.is_available()), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    logger.info("Rank {} initialization finished.".format(global_rank))
    try:
        dist.init_process_group(backend=backend, init_method=dist_url, world_size=world_size, rank=global_rank, timeout=timeout)
    except Exception:
        logger.error("Process group URL: {}".format(dist_url))
        raise

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    main_func(*args)


if __name__ == "__main__":
    args = make_parser().parse_args() # parser模块实例化
    exp = Exp() # 参数控制类负责存储参数,调用网络架构&训练集加载器&测试集加载器&验证集加载器&优化器&学习率调整器,包含预处理等其他函数
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()
    dist_url = "auto" if args.dist_url is None else args.dist_url # 并行训练起始地址
    world_size = args.num_machines * num_gpu # 计算GPU的总数量,机器数量×每台机器的GPU数量
    
    if world_size > 1: # 有卡就并行训练
        if dist_url == "auto":
            assert (args.num_machines == 1), "dist_url=auto cannot work with distributed training."
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", 0))
            port = sock.getsockname()[1]
            sock.close()
            dist_url = f"tcp://127.0.0.1:{port}"

        start_method = "spawn"
        cache = vars(args[1]).get("cache", False)

        # To use numpy memmap for caching image into RAM, we have to use fork method
        if cache:
            assert sys.platform != "win32", ("As Windows platform doesn't support fork method, "
                                             "do not add --cache in your training command.")
            start_method = "fork"

        mp.start_processes(_distributed_worker,
                           nprocs=num_gpu,
                           arg=(main, world_size, num_gpu, args.machine_rank, args.dist_backend, dist_url, (exp, args)),
                           daemon=False,
                           start_method=start_method)
    else:
        main(exp, args) # 没卡就用CPU
