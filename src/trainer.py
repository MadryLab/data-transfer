import os

from threading import Lock
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from optimizers import get_optimizer_and_lr_scheduler

lock = Lock()

class AverageMeter():
    def __init__(self):
        self.num = 0
        self.tot = 0

    def update(self, val, sz):
        self.num += val*sz
        self.tot += sz

    def calculate(self):
        return self.num/self.tot

class ClassAverageMeter(AverageMeter):
    def __init__(self):
        super().__init__()

    def calculate(self):
        return torch.nan_to_num(self.num/self.tot, nan=0.0)

def save_model(model, path, run_metadata):
    torch.save({
        'state_dict': model.state_dict(),
        'run_metadata': run_metadata
    }, path)

def load_model(path, model):
    out = torch.load(path)
    model.load_state_dict(out['state_dict'])
    print(out['run_metadata'])
    return model


class LightWeightTrainer():
    def __init__(self, training_args, ds_name, exp_name, res_args=None, enable_logging=True, granularity="global"):
        self.training_args = training_args
        self.check_training_args_()

        self.prog_res = False
        if res_args is not None:
            self.res_args = res_args
            self.prog_res = True

        self.ds_name = ds_name
        if self.ds_name == "CHESTXRAY14":
            c0 = 74763
            c1 = 3705
            w0 = c1 / (c0 + c1)
            w1 = c0 / (c0 + c1)
            weight = torch.as_tensor([w0, w1]).to("cuda")
            self.ce_loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=training_args['label_smoothing'])
        else:
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=training_args['label_smoothing'])

        self.enable_logging = enable_logging
        if self.enable_logging:
            new_path = self.make_training_dir(exp_name)
            self.training_dir = new_path
            self.writer = SummaryWriter(new_path)

        assert granularity in ["global", "per_class"]
        if granularity == "global":
            self.per_class = False
        else:
            self.per_class = True

    def check_training_args_(self):
        for z in ['epochs', 'lr', 'weight_decay', 'momentum', 'lr_scheduler',
                'step_size', 'gamma', 'lr_milestones', 'lr_peak_epoch', 'eval_epochs']:
            assert z in self.training_args, f'{z} not in training_args'

    def make_training_dir(self, exp_name):
        path = os.path.join('runs', exp_name)
        os.makedirs(path, exist_ok=True)
        existing_count = -1

        for f in os.listdir(path):
            if f.startswith('version_'):
                version_num = f.split('version_')[1]
                if version_num.isdigit() and existing_count < int(version_num):
                    existing_count = int(version_num)
        version_num = existing_count + 1
        new_path = os.path.join(path, f"version_{version_num}")
        print("logging in ", new_path)
        os.makedirs(new_path)
        os.makedirs(os.path.join(new_path, 'checkpoints'))
        return new_path

    def get_resolution(self, epoch):
        min_res, max_res, end_ramp, start_ramp = self.res_args['min_res'], self.res_args['max_res'], self.res_args['end_ramp'], self.res_args['start_ramp']
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    def get_accuracy(self, logits, target):
        correct = logits.argmax(-1) == target
        return (correct.float().mean()) * 100

    def get_accuracy_per_class(self, logits, target):
        cm = torch.zeros(logits.size(1), logits.size(1)).to(target.device)
        preds = logits.max(1)[1]
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
        class_acc = cm.diag()/cm.sum(1)
        class_acc = torch.nan_to_num(class_acc, nan=0.0)
        class_size = cm.sum(1)
        return 100*class_acc, class_size

    def get_opt_scaler_scheduler(self, model, iters_per_epoch=1):
        opt, scheduler = get_optimizer_and_lr_scheduler(self.training_args, model, iters_per_epoch)
        self.per_epoch_lr_scheduler = self.training_args['lr_scheduler'] !=  'cyclic'

        scaler = GradScaler()
        return opt, scaler, scheduler

    def training_step(self, model, batch):
        x, y = batch
        with lock:
            logits = model(x)
        loss = self.ce_loss(logits, y)
        acc = self.get_accuracy(logits, y)
        if self.per_class:
            class_acc, class_size = self.get_accuracy_per_class(logits, y)
        else:
            class_acc, class_size = 0, 0

        return loss, acc, len(x), class_acc, class_size

    def validation_step(self, model, batch):
        x, y = batch
        with lock:
            logits = model(x)
        loss = self.ce_loss(logits, y)
        acc = self.get_accuracy(logits, y)
        if self.per_class:
            class_acc, class_size = self.get_accuracy_per_class(logits, y)
        else:
            class_acc, class_size = 0, 0

        return loss, acc, len(x), class_acc, class_size

    def train_epoch(self, epoch_num, model, train_dataloader, opt, scaler, scheduler):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        with tqdm(train_dataloader) as t:
            t.set_description(f"Train Epoch: {epoch_num}")
            for batch in t:
                opt.zero_grad(set_to_none=True)
                with autocast():
                    loss, acc, sz, class_acc, class_size = self.training_step(model, batch)
                t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
                loss_meter.update(loss.item(), sz)
                acc_meter.update(acc.item(), sz)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if not self.per_epoch_lr_scheduler:
                    scheduler.step()
            if self.per_epoch_lr_scheduler:
                scheduler.step()
        avg_loss, avg_acc = loss_meter.calculate(), acc_meter.calculate()
        return avg_loss, avg_acc

    def val_epoch(self, epoch_num, model, val_dataloader):
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        if self.per_class:
            class_acc_meter = AverageMeter()
        with torch.no_grad():
            with tqdm(val_dataloader) as t:
                t.set_description(f"Val Epoch: {epoch_num}")
                for batch in t:
                    with autocast():
                        loss, acc, sz, class_acc, class_size = self.validation_step(model, batch)
                    t.set_postfix({'loss': loss.item(), 'acc': acc.item()})
                    loss_meter.update(loss.item(), sz)
                    acc_meter.update(acc.item(), sz)
                    if self.per_class:
                        class_acc_meter.update(class_acc, class_size)
        avg_loss, avg_acc = loss_meter.calculate(), acc_meter.calculate()
        if self.per_class:
            avg_class_acc = class_acc_meter.calculate().cpu().detach().numpy()
        else:
            avg_class_acc = 0

        return avg_loss, avg_acc, avg_class_acc


    def fit(self, model, train_dataloader, val_dataloader):
        opt, scaler, scheduler = self.get_opt_scaler_scheduler(model, iters_per_epoch=len(train_dataloader))
        best_val_loss = np.inf
        for epoch in range(self.training_args['epochs']):

            if self.prog_res:
                res = self.get_resolution(epoch)
                train_dataloader.pipelines["image"].operations[0].output_size = (res, res)

            train_loss, train_acc = self.train_epoch(epoch, model, train_dataloader, opt, scaler, scheduler)
            curr_lr = scheduler.get_last_lr()[0]

            is_val_epoch = (epoch % self.training_args['eval_epochs'] == 0 and epoch != 0) or (epoch == self.training_args['epochs'] - 1)

            if is_val_epoch:
                val_loss, val_acc, val_class_acc = self.val_epoch(epoch, model, val_dataloader)
                if self.per_class:
                    print(f"LR: {curr_lr}, Train Loss: {train_loss:0.4f}, Train Acc: {train_acc:0.4f}, Val Loss: {val_loss:0.4f}, Val Acc: {val_acc:0.4f}, Val MeanPerClass: {val_class_acc.mean():0.4f}")
                else:
                    print(f"LR: {curr_lr}, Train Loss: {train_loss:0.4f}, Train Acc: {train_acc:0.4f}, Val Loss: {val_loss:0.4f}, Val Acc: {val_acc:0.4f}")
            else:
                print(f"LR: {curr_lr}, Train Loss: {train_loss:0.4f}, Train Acc: {train_acc:0.4f}")

            # Save Checkpoints
            if self.enable_logging:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Acc/train", train_acc, epoch)
                self.writer.add_scalar("lr", curr_lr, epoch)

                if not is_val_epoch:
                    continue

                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Acc/val", val_acc, epoch)


                if self.per_class:
                    self.writer.add_scalar("MeanPerClass/val", val_class_acc.mean(), epoch)
                    self.writer.add_scalars("Class Acc/val",
                                            {str(k):v for k,v in enumerate(val_class_acc)},
                                            epoch
                    )

                run_metadata = {
                    'training_args': self.training_args,
                    'epoch': epoch,
                    'training_metrics': {'loss': train_loss, 'acc': train_acc},
                    'val_metrics': {'loss': val_loss, 'acc': val_acc},
                }

                if self.per_class:
                    run_metadata['val_metrics'].update({'class_acc': val_class_acc})

                checkpoint_folder = os.path.join(self.training_dir, 'checkpoints')
                checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_latest.pt')
                save_model(model, checkpoint_path, run_metadata)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_best.pt')
                    save_model(model, checkpoint_path, run_metadata)
                if epoch % 5 == 0: # flush every 5 steps
                    self.writer.flush()


