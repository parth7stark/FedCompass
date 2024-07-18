'''
Write code to split the dataset into n chunks. 
n - number of clients

Return dataset chunk according to client id
'''

import os
import torch
import pathlib
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from appfl.config import *
from torch.utils import data
from typing import List, Optional
from appfl.misc.data import Dataset

def get_wavedata(
    num_clients: int,
    client_id: int,
    dataset_dir: str,
    **kwargs
):
    """
    Return the GW dataset chunk for a given client.
    :param num_clients: total number of clients
    :param client_id: the client id
    """

    # Partition the dataset -- wrote another file to partition
#    dataset_dir = "/path/to/dataset_dir/"
    train_file = f"{dataset_dir}train_300_partition_client_{client_id}.hdf5"
    test_file = f"{dataset_dir}test_300_partition_client_{client_id}.hdf5"


    return train_file, test_file
         
    # return train_datasets[client_id], test_dataset