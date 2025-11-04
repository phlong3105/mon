from deband_trainer import IndiDeband
from models.unet2d import  AutoFFTime2d
import torch.nn as nn
from deband_dataset import imageDebandDataset
from torch.utils.data import DataLoader
from utils  import Loss, SetupTrain, getLogger, getModelCallback, set_lightning_seed, fft_2d
import lightning as L

# set training datasets
train = imageDebandDataset(csv='banded-train.csv')
test  = imageDebandDataset(csv='banded-test.csv', test=True)

# Configure loss function, run INDI in the FFT domain, in both directions
loss = Loss("L1 Loss", nn.L1Loss(), transform=fft_2d)

# can replace network here.
net = AutoFFTime2d(
    blocks         = 4,
    in_channels    = 24,
    channel_factor = 24,
    weight         = True,
    neck           = False,
    layout         = 3,
    encoder_dil    = 1
)

# SetupTrain captures all parser args
setup = SetupTrain(net)
set_lightning_seed(setup)
trainLoader = DataLoader(train, batch_size=setup.args.batchSize, num_workers=setup.args.workers, shuffle=True)  # , sampler = sampler )
testLoader  = DataLoader(test,  batch_size=1, num_workers=setup.args.workers, shuffle=True)
denoiser    = IndiDeband(steps=30, net=net, setup=setup, loss=loss)

# get tensorboard logger
logger = getLogger(setup)
checkpoint_callback = getModelCallback(setup)

trainer = L.Trainer(
    max_epochs  = setup.args.epochs,
    logger      = logger,
    devices     = [setup.args.device_id],
    callbacks   = [checkpoint_callback],
    accelerator = setup.device,
    precision   = setup.args.precision
)

# train!
trainer.fit(model=denoiser, train_dataloaders=trainLoader, val_dataloaders=testLoader)
