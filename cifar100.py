from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch import nn
import torch
import cv2
from torch_optimizer import Lookahead, RAdam
from skimage.transform import AffineTransform
import numpy as np
import pickle

NUM_WORKERS = 0


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
        for idx, img in enumerate(imgs):

            if idx > 0:
                alpha = np.random.uniform(224/32*0.8, 224/32*1.2)
                M = AffineTransform(scale=(alpha, alpha))
                img = cv2.warpPerspective(img, M.params, dsize=(224, 224), flags=cv2.INTER_LINEAR)
            else:
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

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
    def __init__(self, alpha=0.8, n: int = 3):
        super().__init__()
        self.alpha = alpha
        self.n = n
        self._enet = EfficientNet.from_name('efficientnet-b0', num_classes=100)
        self._trainval = pickle.loads(open('cifar-100-python/train', 'rb').read(), encoding='bytes')
        self._test = pickle.loads(open('cifar-100-python/test', 'rb').read(), encoding='bytes')

        idx = [i for i in range(0, len(self._trainval[b'filenames']))]
        np.random.seed(0)
        np.random.shuffle(idx)
        cutoff = int(0.8*len(idx))

        self._trainidx = idx[:cutoff]
        self._valididx = idx[cutoff:]

    def forward(self, x, sub_w: float = 1.):
        return self._enet.forward(x, sub_w=sub_w)

    def training_step(self, batch, batch_nb):
        x, x_n, y_target = batch
        # full network
        y = self.forward(x)
        loss = F.cross_entropy(y, y_target)

        loss.backward()
        y_soft = F.softmax(y.detach(), dim=-1)

        # n sub-networks
        for n in range(0, self.n):
            y_sub = self.forward(x_n[n], sub_w=np.random.uniform(self.alpha, 1.))
            y_sub = F.log_softmax(y_sub, dim=-1)
            loss_n = nn.KLDivLoss(reduction='batchmean')(y_sub, y_soft)
            loss_n.backward()

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def backward(self, trainer, loss: torch.Tensor, optimizer: torch.optim.Optimizer, optimizer_idx: int) -> None:
        pass

    def validation_step(self, batch, batch_nb):
        x, _, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, _, y = batch
        y_hat = self(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(),
                                   lr=0.001,
                                   momentum=0.,
                                   weight_decay=1e-4,
                                   eps=1e-5)

    def train_dataloader(self):
        return DataLoader(Cifar100Dataset(self._trainval),
                          sampler=SubsetRandomSampler(self._trainidx),
                          batch_size=32,
                          drop_last=True,
                          num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(Cifar100Dataset(self._trainval, n=0),
                          sampler=SubsetRandomSampler(self._valididx),
                          batch_size=32,
                          num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(Cifar100Dataset(self._test, n=0),
                          batch_size=32,
                          num_workers=NUM_WORKERS)


def main():
    model = Cifar100EfficientNetModule(alpha=0.8, n=3)
    trainer = Trainer(gpus=1, precision=32)
    trainer.fit(model)


if __name__ == '__main__':
    main()