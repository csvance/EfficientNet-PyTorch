import pickle

import cv2
import numpy as np
from skimage.transform import AffineTransform

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torch_optimizer import RAdam, Lookahead

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer

from efficientnet_pytorch import EfficientNet

NUM_WORKERS = 0
EPOCHS = 12
BATCH_SIZE = 32
MAX_LR = 0.01
WEIGHT_DECAY = 1e-4


class Cifar100Dataset(Dataset):
    def __init__(self, data, n: int = 3):
        self.data = data
        self.mean = np.array([0.5071, 0.4865, 0.4409])
        self.std = np.array([0.2673, 0.2564, 0.2762])
        self.n = n

    def __len__(self):
        return len(self.data[b'filenames'])

    def __getitem__(self, item):
        img = np.reshape(self.data[b'data'][item], (3, 32, 32)).transpose((1, 2, 0))

        imgs = [img] + [img for _ in range(0, self.n)]
        x_n = []

        # Augment full network image
        theta = np.deg2rad(np.random.uniform(-30, 30))
        alpha = np.random.uniform(224 / 32 * 0.9, 224 / 32 * 1.1)

        c = alpha * 32 / 2

        M = np.eye(3)
        M = np.matmul(AffineTransform(translation=(-32 / 2, -32 / 2)).params, M)
        M = np.matmul(AffineTransform(rotation=theta, scale=(alpha, alpha)).params, M)
        M = np.matmul(AffineTransform(translation=(c, c)).params, M)
        Mi = M

        for idx, img in enumerate(imgs):

            if idx > 0:
                # Additional scale augmentation for sub networks
                alpha = np.random.uniform(0.9, 1.1)

                Mi = M
                # Center the scale change
                Mi = np.matmul(AffineTransform(translation=(-c, -c)).params, Mi)
                Mi = np.matmul(AffineTransform(scale=(alpha, alpha)).params, Mi)
                Mi = np.matmul(AffineTransform(translation=(c, c)).params, Mi)

            img = cv2.warpPerspective(img, Mi, dsize=(224, 224), flags=cv2.INTER_LINEAR)

            # Normalize
            img = img.astype(np.float64) / 255
            img = (img - self.mean) / self.std
            img = img.astype(np.float32)
            img = img.transpose((2, 0, 1))
            x_n.append(img)

        # Original image and n augmentations
        x = x_n[0]
        x_n = x_n[1:]

        # The label
        y = self.data[b'fine_labels'][item]

        return x, x_n, y


class Cifar100EfficientNetModule(LightningModule):
    def __init__(self, alpha=0.8, n: int = 3, efficientnet: str = 'efficientnet-b0'):
        super().__init__()
        self.alpha = alpha
        self.n = n
        self._enet = EfficientNet.from_name(efficientnet, num_classes=100)
        self._trainval = pickle.loads(open('cifar-100-python/train', 'rb').read(), encoding='bytes')
        self._test = pickle.loads(open('cifar-100-python/test', 'rb').read(), encoding='bytes')

        idx = [i for i in range(0, len(self._trainval[b'filenames']))]
        np.random.seed(0)
        np.random.shuffle(idx)
        cutoff = int(0.8 * len(idx))

        self._trainidx = idx[:cutoff]
        self._valididx = idx[cutoff:]
        self._idx = idx

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x, sub_w: float = 1.):
        return self._enet.forward(x, sub_w=sub_w)

    def training_step(self, batch, batch_nb):
        x, x_n, y_target = batch

        # full network
        y = self.forward(x)
        loss = F.cross_entropy(y, y_target)
        acc_step = self.accuracy(y, y_target)
        self.log('train_acc', acc_step, prog_bar=True, logger=True)
        self.log('train_loss', loss, prog_bar=False, logger=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        self.log('momentum', self.optimizers().param_groups[0]['betas'][0])

        loss.backward()
        y_soft = F.softmax(y.detach(), dim=-1)

        # n sub-networks
        for n in range(0, self.n):
            y_sub = self.forward(x_n[n], sub_w=np.random.uniform(self.alpha, 1.))
            y_sub = F.log_softmax(y_sub, dim=-1)
            loss_n = nn.KLDivLoss(reduction='batchmean')(y_sub, y_soft)
            loss_n.backward()

        return {'loss': loss}

    def training_epoch_end(self, outs):
        self.accuracy.reset()

    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, optimizer_idx: int, *args,
                 **kwargs) -> None:
        pass

    def test_step(self, batch, batch_nb):
        x, _, y_target = batch
        y = self(x)

        loss = F.cross_entropy(y, y_target)

        self.log('test_acc', self.accuracy(y, y_target), prog_bar=True, logger=False)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        self.log('test_acc', self.accuracy.compute(), prog_bar=False, logger=True)
        self.log('test_loss', avg_loss, prog_bar=False, logger=True)

        self.accuracy.reset()

        return {'test_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = Lookahead(RAdam(self.parameters(),
                                    lr=0.001,
                                    weight_decay=WEIGHT_DECAY,
                                    eps=1e-5))
        schedule = {'scheduler': OneCycleLR(optimizer,
                                            max_lr=MAX_LR,
                                            epochs=EPOCHS,
                                            steps_per_epoch=int(len(self._idx) / BATCH_SIZE),
                                            verbose=False),
                    'name': 'learning_rate',
                    'interval': 'step',
                    'frequency': 1
                    }

        return [optimizer], [schedule]

    def train_dataloader(self):
        return DataLoader(Cifar100Dataset(self._trainval),
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          drop_last=True,
                          num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(Cifar100Dataset(self._test, n=0),
                          batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS)


def main():
    model = Cifar100EfficientNetModule(alpha=0.8, n=3,
                                       efficientnet='efficientnet-b0')

    trainer = Trainer(gpus=1,
                      precision=32,
                      max_epochs=EPOCHS,
                      log_every_n_steps=5,
                      flush_logs_every_n_steps=10)
    trainer.fit(model)
    trainer.save_checkpoint('checkpoints/cifar-100.ckpt')
    trainer.test()


if __name__ == '__main__':
    main()
