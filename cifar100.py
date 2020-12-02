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

# p is the amount of padding on each side of the image
P = 2
# sz is the size of the image before padding
SZ = 32
# szt is the target size
SZT = 224


def affine_about_recenter(center, recenter, rotation, scale):
    M = AffineTransform(translation=(-center[0], -center[1])).params
    M = np.matmul(AffineTransform(rotation=rotation, scale=(scale, scale)).params, M)
    M = np.matmul(AffineTransform(translation=(recenter[0], recenter[1])).params, M)
    return M


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

        import matplotlib.pyplot as plt
        plt.imshow(cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR))
        plt.show()

        # Pad the image so we we don't have dark areas at the border
        img = cv2.copyMakeBorder(img, P, P, P, P, borderType=cv2.BORDER_REPLICATE)

        imgs = [img] + [img for _ in range(0, self.n)]
        x_n = []

        # Compensate for padding
        M = AffineTransform(translation=(-P, -P)).params

        if self.n > 0:
            # If n > 0 then we assume we are in training mode so we use augment each image
            theta = np.deg2rad(np.random.uniform(-30, 30))
            alpha = np.random.uniform(0.9, 1.1)*SZT/SZ
            M = np.matmul(affine_about_recenter(center=(SZ / 2, SZ / 2),
                                                recenter=(SZT / 2 + 3, SZT / 2 + 3),
                                                rotation=theta,
                                                scale=alpha),
                          M)
        elif self.n == 0:
            # If n == 0 we are in evaluation mode so don't apply augmentation
            alpha = SZT/SZ
            M = np.matmul(affine_about_recenter(center=(SZ / 2, SZ / 2),
                                                recenter=(SZT / 2 + 3, SZT / 2 + 3),
                                                rotation=0,
                                                scale=alpha),
                          M)
        else:
            raise ValueError

        for idx, img in enumerate(imgs):
            Mi = M

            if idx > 0:
                # Additional scale augmentation for sub networks
                alpha = np.random.uniform(0.9, 1.1)
                Mi = np.matmul(affine_about_recenter((SZT/2, SZT/2), (SZT/2, SZT/2), 0, alpha), M)

            img = cv2.warpPerspective(img, Mi, dsize=(SZT, SZT), flags=cv2.INTER_LINEAR)

            plt.imshow(img)
            plt.show()

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
                                            steps_per_epoch=int(len(self._trainval[b'filenames']) / BATCH_SIZE),
                                            verbose=False),
                    'name': 'learning_rate',
                    'interval': 'step',
                    'frequency': 1
                    }

        return [optimizer], [schedule]

    def train_dataloader(self):
        return DataLoader(Cifar100Dataset(self._trainval, n=self.n),
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
