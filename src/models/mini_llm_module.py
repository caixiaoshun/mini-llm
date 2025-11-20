from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup


class MiniLLMLitModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        compile: bool,
        lr,
        warmup_ratio,
        aux_loss_weight=0.05,
        min_lr_rate=0.1,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=[net])

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(
            task="multiclass", num_classes=32000, ignore_index=-100
        )
        self.val_acc = Accuracy(task="multiclass", num_classes=32000, ignore_index=-100)
        self.test_acc = Accuracy(
            task="multiclass", num_classes=32000, ignore_index=-100
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        aux_loss = None
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = self.forward(input_ids)
        if isinstance(logits, (tuple, list)):
            logits, aux_loss = logits
        batch_size, seq_len, dim = logits.shape
        logits = logits.reshape(batch_size * seq_len, -1)
        labels = labels.reshape(
            -1,
        )

        loss = self.criterion(logits, labels)
        if aux_loss is not None:
            loss += aux_loss * self.hparams.aux_loss_weight

        preds = torch.argmax(logits, dim=-1)
        return loss, preds, labels

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train_loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log(
            "val_loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer = torch.optim.AdamW(
            params=self.net.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        # 计算总训练步数（已包含梯度累计/采样比例/limit_*_batches 等）
        total_steps = self.trainer.estimated_stepping_batches

        warmup_ratio = self.hparams.warmup_ratio

        warmup_steps = max(1, int(total_steps * float(warmup_ratio)))

        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_rate=self.hparams.min_lr_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 每个 step 调一次
                "frequency": 1,
            },
        }
