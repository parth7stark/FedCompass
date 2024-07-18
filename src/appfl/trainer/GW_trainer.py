import copy
import time
import torch
import importlib
import numpy as np
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, Any
from torch.utils.data import Dataset, DataLoader
from appfl.trainer.base_trainer import BaseTrainer
from appfl.privacy import laplace_mechanism_output_perturb

# from __future__ import print_function
import numpy as np
from time import time
import os
import sys
import gc
#import cProfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import CSVLogger

# from models_torch import *
# Current working directory: /scratch/bblq/parthpatel7173/APPFL/FedCompass/examples
# Add the directory containing data_generators_torch.py to the sys.path
sys.path.append('/scratch/bblq/parthpatel7173/APPFL/FedCompass/src/appfl/trainer/')

from data_generators_torch import *
# from custom_callbacks_torch import *

class LightningModel(LightningModule):
    def __init__(self, model, lr):
        super(LightningModel, self).__init__()

        # Instantiate the model -- model architecture/code present in models_torch.py
        # self.model = full_module()
        self.model = model
        # self.loss_fn = loss_fn
        # self.metric = metric
        self.lr = lr
        
        # Move the model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def forward(self, x):
        return self.model(x)

    # Define the optimizer and learning rate scheduler
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-10, verbose=True),
                'monitor': 'val_loss',  
                'interval': 'epoch',  
                'frequency': 1,  
                'strict': True,
            }
        }

    def training_step(self, batch, batch_idx):
        inputs, targets = batch  
        outputs = self.model(inputs)
        loss = F.binary_cross_entropy(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch  
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = F.binary_cross_entropy(outputs, targets)
        #store it temp file
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss
        
    def on_train_epoch_end(self, unused=None):  
        # Increment the epoch counter of training dataset at the end of each epoch
        self.trainer.datamodule.train_dataset.increment_epoch()
        

class GWTrainer(BaseTrainer):
    """
    GWTrainer:
        Trainer for FL clients, which trains the model using `torch.optim` 
        optimizers for a certain number of local epochs or local steps. 
        Users need to specify which training model to use in the configuration, 
        as well as the number of local epochs or steps.
    """  
    def __init__(
        self,
        model: Optional[Module]=None,
        loss_fn: Optional[Module]=None,
        metric: Optional[Any]=None,
        train_dataset: Optional[Dataset]=None,
        val_dataset: Optional[Dataset]=None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any]=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs
        )

        if not hasattr(self.train_configs, "device"):
            self.train_configs.device = "cpu"

        self._sanity_check()

        # #Instantiate the trainer in init() and call model.fit() in train()
        # client_name = self.train_configs.logging_id
        # #callbacks
        # RegularModelCheckpoints=ModelCheckpoint(dirpath=self.train_configs.checkpoint_dir,filename=client_name + '_model_lre-3_{epoch:02d}-{val_loss:.5f}',monitor='val_loss',mode='min',save_top_k=-1)
        
        # StopCriteria=EarlyStopping(monitor='val_loss',patience=7, verbose=True, mode='min')
        
        # callbacks=[RegularModelCheckpoints,LearningRateMonitor(logging_interval='epoch'),]
        

        # # Determine the number of available GPU devices
        # # devices=torch.cuda.device_count()
        # # devices=devices if devices!=0 else 4 

        # # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
        # devices=1 if self.train_configs.device == "cpu" else torch.cuda.device_count()
        
        # accelerator="cpu" if self.train_configs.device == "cpu" else "gpu"
        
        # # Configure the logger
        # # csv_logger = CSVLogger("./logs", name="my_model")
        # csv_logger = CSVLogger("./logs", name=self.train_configs.logging_id, version=f"global_epoch_{self.round}"


        # # self.logger = TensorBoardLogger(
        # #         "logs",
        # #         name=self.train_configs.logging_id,
        # #         version=f"version_{self.train_configs.logging_id}"
        # #     )
        # # define trainer
        # self.trainer = Trainer(
        #     max_epochs=self.train_configs.num_local_epochs,
        #     num_nodes=self.train_configs.num_nodes,devices=devices,accelerator=accelerator, strategy="ddp",     
        #     enable_progress_bar=False,
        #     enable_model_summary=True,
        #     callbacks=callbacks,
        #     logger=csv_logger,
        # )
        #     # logger=self.logger,

        #             # log_every_n_steps=1,


        
    def train(self):
        """
        Train the model for a certain number of local epochs or steps and store the mode state
        (probably with perturbation for differential privacy) in `self.model_state`.
        """

        #Instantiate the trainer in init() and call model.fit() in train()
        client_name = self.train_configs.logging_id
        #callbacks
        RegularModelCheckpoints=ModelCheckpoint(dirpath=self.train_configs.checkpoint_dir,filename=client_name + '_model_lre-3_{epoch:02d}-{val_loss:.5f}',monitor='val_loss',mode='min',save_top_k=-1)
        
        StopCriteria=EarlyStopping(monitor='val_loss',patience=7, verbose=True, mode='min')
        
        callbacks=[RegularModelCheckpoints,LearningRateMonitor(logging_interval='epoch'),]
        

        # Determine the number of available GPU devices
        # devices=torch.cuda.device_count()
        # devices=devices if devices!=0 else 4 

        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
        devices=1 if self.train_configs.device == "cpu" else torch.cuda.device_count()
        
        accelerator="cpu" if self.train_configs.device == "cpu" else "gpu"
        
        # Configure the logger
        csv_logger = CSVLogger("./logs", name=self.train_configs.logging_id, version=f"global_epoch_{self.round}")

        # define trainer
        self.trainer = Trainer(
            max_epochs=self.train_configs.num_local_epochs,
            num_nodes=self.train_configs.num_nodes,
            devices=devices,accelerator=accelerator, strategy="ddp",     
            enable_progress_bar=False,
            enable_model_summary=True,
            callbacks=callbacks,
            logger=csv_logger,
        )

        #define model
        model = LightningModel(self.model, lr=self.train_configs.optim_args.lr)
        
        # Load Data Generators - code present in data_generators_torch.py
        '''
        need to modify above file to load dataset given client dataset file name
        '''
        self.data_module = WaveformDataModule(self.train_configs.noise_dir, train_file=self.train_dataset, val_file=self.val_dataset, batch_size=self.train_configs.get("train_batch_size", 32),n_channels=self.train_configs.n_channels,gaussian=1,noise_prob=0.7, noise_range=None, num_workers=self.train_configs.get("num_workers", 0))

        # train
        t0 = time()
        self.trainer.fit(model,datamodule=self.data_module)
        t1 = time()

        print('**Evaluation time: %s' % (t1 - t0))
        #read val loss from temp_file
        self.round += 1
        # Store the previous model state for gradient computation
        # send_gradient = self.train_configs.get("send_gradient", False)
        # if send_gradient:
        #     self.model_prev = copy.deepcopy(self.model.state_dict())

        # self.model.to(self.train_configs.device)

        # self.val_dataloader=data.val_dataloader()

        # do_validation = self.train_configs.get("do_validation", False) and self.val_dataloader is not None
        # do_pre_validation = self.train_configs.get("do_pre_validation", False) and do_validation
        
        # # Set up logging title
        # if self.round == 0:
        #     title = (
        #         ["Round", "Time", "Train Loss", "Train Accuracy"] 
        #         if not do_validation
        #         else (
        #             ["Round", "Pre Val?", "Time", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"] 
        #             if do_pre_validation 
        #             else ["Round", "Time", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"]
        #         )
        #     )
        #     if self.train_configs.mode == "epoch":
        #         title.insert(1, "Epoch")
        #     self.logger.log_title(title)

        # if do_pre_validation:
        #     val_loss, val_accuracy = self._validate()
        #     content = [self.round, "Y", " ", " ", " ", val_loss, val_accuracy]  
        #     if self.train_configs.mode == "epoch":
        #         content.insert(1, 0)
        #     self.logger.log_content(content)
        
            # train
        # t0 = time()
        # trainer.fit(model,datamodule=data_module)
        # t1 = time()
       
        # Differential privacy -- cant leave for demo
        if self.train_configs.get("use_dp", False):
            assert hasattr(self.train_configs, "clip_value"), "Gradient clipping value must be specified"
            assert hasattr(self.train_configs, "epsilon"), "Privacy budget (epsilon) must be specified"
            sensitivity = 2.0 * self.train_configs.clip_value * self.train_configs.optim_args.lr
            self.model_state = laplace_mechanism_output_perturb(
                self.model,
                sensitivity,
                self.train_configs.epsilon,
            )
        else:
            self.model_state = copy.deepcopy(self.model.state_dict())
        
        # Move to CPU for communication
        if self.train_configs.device == "cuda":
            for k in self.model_state:
                self.model_state[k] = self.model_state[k].cpu()


    def get_parameters(self) -> Dict:
        hasattr(self, "model_state"), "Please make sure the model has been trained before getting its parameters"
        return self.model_state

    def _sanity_check(self):
        """
        Check if the configurations are valid.
        """
        assert hasattr(self.train_configs, "mode"), "Training mode must be specified"
        assert self.train_configs.mode in ["epoch", "step"], "Training mode must be either 'epoch' or 'step'"
        if self.train_configs.mode == "epoch":
            assert hasattr(self.train_configs, "num_local_epochs"), "Number of local epochs must be specified"
        else:
            assert hasattr(self.train_configs, "num_local_steps"), "Number of local steps must be specified"

    def validate_model(self) -> Tuple[float, int]:
        """
        Validate the model on the validation dataset and return the validation loss and sample size.
        """
        self.model.eval()  # Set the model to evaluation mode
        
        val_loss = 0.0
        val_samples = 0
        
        val_dataloader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.to("cuda"), targets.to("cuda")
                
                outputs = self.model(inputs)
                loss = F.binary_cross_entropy(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
        
        val_loss /= val_samples
        return val_loss, val_samples
    

    # def validate_model(self) -> Tuple[float, int]:
    #         """
    #         Validate the model on the validation dataset and return the validation loss and sample size.
    #         """
    #         self.model.eval()  # Set the model to evaluation mode
            
    #         val_loss = 0.0
    #         val_samples = 0
            
    #         val_dataloader = DataLoader(self.val_dataset, batch_size=self.train_configs.get("val_batch_size", 32), shuffle=False)
            
    #         with torch.no_grad():
    #             for inputs, targets in val_dataloader:
    #                 if torch.cuda.is_available():
    #                     inputs, targets = inputs.to("cuda"), targets.to("cuda")
                    
    #                 outputs = self.model(inputs)
    #                 loss = F.binary_cross_entropy(outputs, targets)
    #                 val_loss += loss.item() * inputs.size(0)
    #                 val_samples += inputs.size(0)
            
    #         val_loss /= val_samples
    #         return val_loss, val_samples