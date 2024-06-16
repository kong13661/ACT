import copy

import os

import inspect
import fire
import shutil
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, OnExceptionCheckpoint
import warnings
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.image.fid import FrechetInceptionDistance
import wandb

from path_utils import PathAuxiliary
from torchvision.utils import make_grid
warnings.filterwarnings("ignore")
from common import get_params_to_record_from_funcion_signature
from torchvision.utils import save_image
from functools import wraps
from dataset import get_data_module
from torchmetrics import MeanMetric
from models.loss import get_distance_loss
import einops
import math
from models.consistency_wrapper import ConsistencyWrapper
from pytorch_lightning.utilities import rank_zero_only
from models.unet import UNet2DModel
from models.discriminator_cifar10 import UNet2DModelDiscriminator
from augmentation import augment
from torch import autocast
import numpy as np


def grid_and_save(image, name):
    grid = make_grid(image, nrow=8, normalize=True)
    save_image(grid, 'image/' + name + ".png")


def get_model(config):
    if config['model'] == 'unet':
        g = UNet2DModel(
            sample_size=32,
            in_channels=3,
            out_channels=6,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )

    d = UNet2DModelDiscriminator(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=3,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D"
        ),
    )
    if config['unet_compile']:
        g = torch.compile(g)
    return ConsistencyWrapper(g, d, **config['model_config'])


class TrainingStep(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = get_model(config)
        self.model : ConsistencyWrapper
        self.model_ema : ConsistencyWrapper = copy.deepcopy(self.model)
        # del self.model_ema._discriminator

        self.optim = config['optim']
        self.lr = config['lr']
        self.optim = config['optim']
        self.warmup = config['warmup']
        self.distance_loss = get_distance_loss(config['consistency_distance'], config['compile'])

        self._r1_ema = config['r1_threshold']
        self.config = config

        self.consistency_loss = self._interval_wrapper(
            self.consistency_loss, 1, 'consistency_loss')
        self.r1_panelty = self._interval_wrapper(
            self.r1_panelty, config['r1_interval'], 'r1_loss')
        self.generator_loss = self._interval_wrapper(
            self.generator_loss, 1, ('gloss_fx'))
        self.discriminator_fake_loss = self._interval_wrapper(
            self.discriminator_fake_loss, 1, ('d_fake_loss'))
        self.discriminator_real_loss = self._interval_wrapper(
            self.discriminator_real_loss, 1, ('d_real_loss', 'none'))
        self.discriminator_training_step = self._interval_wrapper(
            self.discriminator_training_step, 1, ('dloss_total', 'none'))

        self.sample_size = config['sample_size']
        self._bins_tracker = MeanMetric()
        self._ema_decay_tracker = MeanMetric()
        self._aug_p_tracker = MeanMetric()
        self._r1_ema_tracker = MeanMetric()
        self.ada_aug_p = 0

        self.automatic_optimization = False
        self.total_step = 0

        self.fid = FrechetInceptionDistance(normalize=True, reset_real_features=False)
        if config['compile']:
            self.fid = torch.compile(self.fid)

    def r1_panelty(self, real_logits, real):
        grad_real = torch.autograd.grad(
            outputs=real_logits.sum(), inputs=real, create_graph=True
        )
        grad_penalty = torch.sum(torch.square(grad_real[0]), dim=[1, 2, 3]).mean()
        r1_loss = grad_penalty * (self.config['r1_gamma'] * 0.5)
        self.update_r1_ema_and_aug_p(r1_loss.detach())
        return r1_loss

    def loss_func(self, x, y):
        return torch.nn.functional.softplus(x * y).mean()

    def loss_func_g(self, x, y, t):
        if self.config['consistency_loss_weight'] == 'edm':
            weight = self.model.get_scalings_for_boundary_condition(t)[-1]
        else:
            weight = 1
        return (torch.nn.functional.softplus(x * y) * weight).mean()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "TrainingStep", outputs
    ) -> None:
        self.ema_update()

    def _interval_wrapper(self, func, interval, log_names):
        if not isinstance(log_names, (tuple, list)):
            log_names = (log_names,)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.total_step % interval == 0:
                signiture = inspect.signature(func)
                if 'self' in signiture.parameters:
                    out = func(self, *args, **kwargs)
                else:
                    out = func(*args, **kwargs)
                if not isinstance(out, (tuple, list)):
                    out = (out,)
                for log_n, _out in zip(log_names, out):
                    if log_n != 'none':
                        setattr(self, f'_{log_n}_cache', _out.item())
            else:
                out = (0,) * len(log_names)
            assert len(log_names) == len(out)
            for log_n in log_names:
                if log_n != 'none':
                    self.log(log_n, getattr(self, f'_{log_n}_cache', 0), on_epoch=False, on_step=True, prog_bar=True,
                             sync_dist=True, logger=True)

            if len(out) == 1:
                return out[0]
            return out
        return wrapper

    def training_step(self, batch, batch_idx):
        self.time_max = 82.0
        x, _ = batch
        # x = self.cutout_op(x)
        x_cal_gloss = x[:x.shape[0] // 2]
        x_cal_dloss = x[x.shape[0] // 2:]

        optimizer_g, optimizer_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()
        for i in range(self.config['accumu']):
            idx = i * x_cal_gloss.shape[0] // self.config['accumu']
            idx_r = (i + 1) * x_cal_gloss.shape[0] // self.config['accumu']
            loss_g = self.generator_training_step(x_cal_gloss[idx: idx_r])
            self.manual_backward(loss_g / self.config['accumu'])
        optimizer_g.step()
        scheduler_g.step()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad()
        for i in range(self.config['accumu']):
            idx = i * x_cal_dloss.shape[0] // self.config['accumu']
            idx_r = (i + 1) * x_cal_dloss.shape[0] // self.config['accumu']
            *_, loss_d = self.discriminator_training_step(x_cal_dloss[idx: idx_r])
            self.manual_backward(loss_d / self.config['accumu'])
        optimizer_d.step()
        scheduler_d.step()
        self.untoggle_optimizer(optimizer_d)

        self.total_step += 1

        self._bins_tracker(self.bins)
        self._r1_ema_tracker(self._r1_ema)
        self.log(
            "bins",
            self._bins_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "r1_ema",
            self._r1_ema_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

    def discriminator_training_step(self, x):
        x1 = x[:x.shape[0] // 2]
        x2 = x[x.shape[0] // 2:]

        self.model.generator_training(False)
        fake_loss_total = self.discriminator_fake_loss(x1)
        real_loss_total, r1_loss = self.discriminator_real_loss(x2)
        dloss_total = (fake_loss_total + real_loss_total)

        return dloss_total, (dloss_total  + r1_loss) * self.config['gan_ratio']

    def discriminator_fake_loss(self, x):
        t = self.model.train_time_sampler(self.bins, x.shape[0])
        noise = torch.randn_like(x)
        xt = self.model.diffusion(x, t, noise)
        with torch.no_grad():
            fake_image = self.model(xt, t)

        fake_image = augment(fake_image, self.ada_aug_p)[0].detach()

        fake_output_fx = self.model.discriminator(fake_image, t)

        fake_loss_fx = self.loss_func(fake_output_fx, torch.ones_like(fake_output_fx))
        return fake_loss_fx

    def discriminator_real_loss(self, x):
        t = self.model.train_time_sampler(self.bins, x.shape[0])
        x = x.clone()
        x.requires_grad = True
        x = augment(x, self.ada_aug_p)[0]
        real_output_fx = self.model.discriminator(x, t)

        real_loss_fx = self.loss_func(real_output_fx, - torch.ones_like(real_output_fx))
        r1_loss_fx = self.r1_panelty(real_output_fx, x) if self.config['r1_gamma'] > 0 else 0

        return real_loss_fx, r1_loss_fx

    def generator_training_step(self, x):
        self.model.generator_training(True)

        next_t = self.model.train_time_sampler(self.bins, x.shape[0])
        sample_t = self.model.prev_time(next_t, self.bins)
        noise = torch.randn_like(x)

        xt = self.model.diffusion(x, sample_t, noise)
        xt_next_noise = self.model.diffusion(x, next_t, noise)

        with torch.no_grad():
            sample_xt = self.model_ema(xt, sample_t)
        sample_mixture = xt_next_noise

        sample_x = self.model(sample_mixture, next_t)

        consistency_loss = self.consistency_loss(sample_x, sample_xt, next_t)
        g_loss = self.generator_loss(sample_x, next_t)

        self._aug_p_tracker(self.ada_aug_p)
        self.log(
            "aug_p",
            self._aug_p_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        return torch.lerp(consistency_loss, g_loss, self.config['gan_ratio'])
        # return g_loss

    def generator_loss(self, fake_image, t):
        if self.config['gan_ratio'] == 0:
            return torch.tensor(0., device=self.device)
        fake_image = augment(fake_image, self.ada_aug_p)[0]
        fake_output_fx = self.model.discriminator(fake_image, t)
        fake_loss_fx = self.loss_func_g(fake_output_fx, - torch.ones_like(fake_output_fx), t)
        return fake_loss_fx

    def consistency_loss(self, target, pred, t):
        if self.config['consistency_loss_weight'] == 'none':
            return self.distance_loss(target, pred, 1).mean()
        elif self.config['consistency_loss_weight'] == 'edm':
            weight = self.model.get_scalings_for_boundary_condition(t)[-1]
            return self.distance_loss(target, pred, weight).mean()

    @property
    def bins(self) -> int:
        if self.config['bin_progress']:
            return round(
                math.sqrt(
                    self.trainer.global_step
                    / self.trainer.estimated_stepping_batches
                    * (self.config['bins_max']**2 - self.config['bins_min']**2)
                    + self.config['bins_min']**2
                )
            )

        return self.config['bins_max']

    @property
    def training_pregress(self):
        return self.trainer.global_step / self.trainer.estimated_stepping_batches

    def configure_optimizers(self):
        def warmup_lr(step):
            return min(step, self.warmup) / self.warmup
        r = 16 / 17
        if self.optim == 'adam':
            optimizer_g = torch.optim.Adam(self.model.unet.parameters(), betas=(0, 0.99), lr=self.lr)
            optimizer_d = torch.optim.Adam(self.model._discriminator.parameters(), betas=(0, 0.99), lr=self.config['d_lr'])

        if self.optim == 'radam':
            optimizer_g = torch.optim.RAdam(self.model.unet.parameters(), lr=self.lr, betas=(0, 0.99))
            optimizer_d = torch.optim.RAdam(self.model._discriminator.parameters(), lr=self.config['d_lr'], betas=(0, 0.99 * r))

        if self.optim == 'RMSprop':
            optimizer_g = torch.optim.RMSprop(self.model.unet.parameters(), lr=self.lr)
            optimizer_d = torch.optim.RMSprop(self.model._discriminator.parameters(), lr=self.config['d_lr'])

        return [optimizer_g, optimizer_d], [torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_lr),
                                            torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_lr)]

    def on_validation_epoch_end(self) -> None:
        fid = self.fid.compute()
        self.fid.reset()
        self.log('fid', fid, on_epoch=True, on_step=False, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        with autocast(device_type='cuda', enabled=False):
            x, _ = batch
            noise = torch.randn_like(x) * self.config['sigma_max']
            x_sample = self.model_ema(noise, x.new_ones([x.shape[0], ], dtype=torch.float32) * self.config['sigma_max'])
            x_sample = x_sample.clip(-1, 1)
            self.fid.update((x + 1) / 2, real=True)
            self.fid.update((x_sample + 1) / 2, real=False)

            if batch_idx == 0:
                x = x[:self.sample_size]
                x_grid = make_grid((x + 1) / 2)

                noise = torch.randn_like(x) * self.config['sigma_max']
                x_sample = self.model_ema(noise, x.new_ones([x.shape[0], ], dtype=torch.float32) * self.config['sigma_max'])
                x_sample = x_sample.clip(-1, 1)
                x_sample_grid = make_grid((x_sample + 1) / 2)

                self.trainer.loggers[0].log_image('dataset', [x_grid], self.trainer.global_step)
                self.trainer.loggers[0].log_image('sample', [x_sample_grid], self.trainer.global_step)

                x, _ = batch
                x = x[:6]
                x: torch.Tensor
                xts = []
                ts = []
                times = [(self.config['bins_max'] - i - 1) for i in range(0, self.config['bins_max'], self.config['bins_max'] // 10)]
                times.sort()
                noise = torch.randn_like(x)
                for t in times:
                    xts.append((x + noise * self.model.timesteps_to_time(x.new_ones([x.shape[0], 1, 1, 1], dtype=torch.float32) * t, self.config['bins_max'])))
                    ts.append(self.model.timesteps_to_time(x.new_ones([x.shape[0], ], dtype=torch.float32) * t, self.config['bins_max']))
                xts = torch.cat(xts, dim=0)
                ts = torch.cat(ts, dim=0)
                xts_sample = self.model_ema(xts, ts)
                xts = xts.reshape(-1, 6, *xts.shape[1:])
                xts = torch.cat([x[None, ...], xts], dim=0).transpose(0, 1)
                xts = xts / einops.reduce(xts, 'b t c h w -> b t 1 1 1', 'max')
                xts_grid = make_grid((xts.reshape(-1, *xts.shape[2:]) + 1) / 2, 11)

                xts_sample = xts_sample.reshape(-1, 6, *xts_sample.shape[1:])
                xts_sample = torch.cat([x[None, ...], xts_sample], dim=0).transpose(0, 1)
                xts_sample = xts_sample.clip(-1, 1)
                xts_sample_grid = make_grid((xts_sample.reshape(-1, *xts_sample.shape[2:]) + 1) / 2, 11)

                self.trainer.loggers[0].log_image('sample_t', [xts_sample_grid], self.trainer.global_step)
                self.trainer.loggers[0].log_image('sample_noise', [xts_grid], self.trainer.global_step)

    def sample(self):
        with autocast(device_type='cuda', enabled=False):
            x = torch.empty([self.config['sample_num'], 3, self.config['image_size'], self.config['image_size']], device=self.device)
            noise = torch.randn_like(x) * self.config['sigma_max']
            x_sample = self.model_ema(noise, x.new_ones([x.shape[0], ], dtype=torch.float32) * self.config['sigma_max'])
            x_sample = x_sample.clip(-1, 1)
            x_sample_grid = make_grid((x_sample + 1) / 2)
            save_image(x_sample_grid, os.path.join(self.config['path_auxiliary'].sample, 'sample.png'))

    @torch.no_grad()
    def ema_update(self):
        param = [p.data for p in self.model.unet.parameters()]
        param_ema = [p.data for p in self.model_ema.unet.parameters()]

        torch._foreach_mul_(param_ema, self.ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay)

        self._ema_decay_tracker(self.ema_decay)
        self.log(
            "ema_decay",
            self._ema_decay_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
        )

    @property
    def ema_decay(self):
        return math.exp(self.config['bins_min'] * math.log(self.config['initial_ema_decay']) / (self.bins - 1))

    def update_r1_ema_and_aug_p(self, r1_loss):
        self._r1_ema = self._r1_ema * self.config['loss_ema_decay'] \
            + r1_loss * (1 - self.config['loss_ema_decay'])

        if self._r1_ema > self.config['r1_threshold']:
            self.ada_aug_p += self.config['batch_size'] * self.config['r1_interval'] / self.config['ada_length']
        else:
            self.ada_aug_p -= self.config['batch_size'] * self.config['r1_interval'] / self.config['ada_length']
        self.ada_aug_p = min(1, max(0, self.ada_aug_p))


def set_wandb(config):
    path_auxiliary: PathAuxiliary = config['path_auxiliary']
    if rank_zero_only.rank != 0:
        return WandbLogger()

    wandb_logger = WandbLogger(group=path_auxiliary.table_name,
                               name="train", save_dir=path_auxiliary.log_wandb, project=path_auxiliary.project_name)
    return wandb_logger


def run(config):
    datamodule, *_ = get_data_module(config)
    path_auxiliary: PathAuxiliary = config['path_auxiliary']
    wandb_logger = set_wandb(config)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=config['device'],
        strategy='ddp_find_unused_parameters_true' if config['device'] != 1 else 'auto',
        enable_progress_bar=True,
        callbacks=[
            OnExceptionCheckpoint(path_auxiliary.checkpoint, "on_exception"),
        ] + ([ModelCheckpoint(path_auxiliary.checkpoint, "-checkpoint-{epoch}-{step}", every_n_train_steps=config['save_every'], save_last=True,
                              save_on_train_epoch_end=False, save_top_k=-1)]),
        logger=[wandb_logger],
        max_steps=config['max_steps'],
        fast_dev_run=config['fast_dev_run'],
        num_sanity_val_steps=0,
        sync_batchnorm=True,
        precision=32,
        val_check_interval=2000,
        check_val_every_n_epoch=None,
    )

    if rank_zero_only.rank == 0:
        trainer.logger.experiment.config.update(get_params_to_record_from_funcion_signature(config, config['function_signature']))
    ckpt_path = config['resume_from']
    # trainer.fit(TrainingStep(config), datamodule=datamodule, ckpt_path=ckpt_path)
    datamodule.setup('fit')
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    if config['mode'] == 'train':
        trainer.fit(TrainingStep(config), train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)
    if config['mode'] == 'sample':
        train_step = TrainingStep(config)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        train_step.load_state_dict(ckpt['state_dict'])
        train_step.cuda()
        train_step.sample()
    if config['mode'] == 'eval':
        trainer.validate(TrainingStep(config), dataloaders=val_dataloader, ckpt_path=ckpt_path)


def main(
    dataset                     : str       = 'cifar10',
    batch_size                  : int       = 80,
    lr                          : float     = 1e-4,
    d_lr                        : float     = 1e-4,
    optim                       : str       = 'radam',
    warmup                      : int       = 1,
    consistency_distance        : str       = 'lpips',
    bin_progress                : bool      = True,
    bins_min                    : int       = 2,
    bins_max                    : int       = 150,
    initial_ema_decay           : float     = 0.9,
    gan_ratio                   : float     = 0.3,
    ada_length                  : float     = 500000,

    loss_ema_decay              : float     = 0.93,
    model                       : str       = 'unet',
    r1_interval                 : int       = 16,
    r1_gamma                    : float     = 10,
    r1_threshold                : float     = 10000,
    layer_per_block             : int       = 2,
    mode                                    = "train",  # train sample eval
    sample_num                              = 64,
    image_size                              = 32,

    sigma_data                              = 0.5,
    sigma_min                               = 0.002,
    sigma_max                               = 80,
    rho                         : float     = 7,
    max_steps                   : int       = 300000,



    save_every                              = 10000,
    accumu                                  = 1,
    sample_size                             = 64,
    device                                  = 1,
    compile                     : bool      = False,
    unet_compile                : bool      = False,  # compile still on develepment (pytorch). It dose not work on multiple gpu.

    dataset_path                            = None,
    ckpt_path                               = None,
    resume_from                             = None,
    fast_dev_run                            = False,
):
    batch_size = batch_size * device * 2  # a half of the batch is used for generator training, and the other half is used for discriminator training
    max_steps = max_steps * 2  # pytorch_lightning treat an optim invoke as a step, so we need to double the max_steps
    consistency_loss_weight: str = 'none'
    t_rescale = False

    path_auxiliary = PathAuxiliary(dataset_path=dataset_path, ckpt_path=ckpt_path, create_path=True, debug=False)
    config = locals()
    config['model_config'] = {
        't_rescale': t_rescale,
        'sigma_data': sigma_data,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'rho': rho,
    }
    config['function_signature'] = inspect.signature(main)
    path_auxiliary.function_signature = config['function_signature']
    run(config)


if __name__ == '__main__':
    fire.Fire(main)
