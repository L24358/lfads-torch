import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from typing import List, Union
from .tuples import SessionBatch, AreaSessionBatch
from .utils import get_paths

MANDATORY_KEYS = {
    "train": ["encod_data", "recon_data"],
    "valid": ["encod_data", "recon_data"],
    "test": ["encod_data"],
}
PATHS = get_paths()

def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


def attach_tensors(datamodule, data_dicts: List[dict], extra_keys: List[str] = []):
    hps = datamodule.hparams
    sv_gen = torch.Generator().manual_seed(hps.sv_seed)
    all_train_data, all_valid_data, all_test_data = [], [], []
    for data_dict in data_dicts:

        def create_session_batch(prefix, extra_keys=[]):
            # Ensure that the data dict has all of the required keys
            assert all(f"{prefix}_{key}" in data_dict for key in MANDATORY_KEYS[prefix])
            # Load the encod_data
            encod_data = to_tensor(data_dict[f"{prefix}_encod_data"])
            n_samps, n_steps, _ = encod_data.shape
            # Load the recon_data
            if f"{prefix}_recon_data" in data_dict:
                recon_data = to_tensor(data_dict[f"{prefix}_recon_data"])
            else:
                recon_data = torch.zeros(n_samps, 0, 0)
            if hps.sv_rate > 0:
                # Create sample validation mask # TODO: Sparse and use complement?
                bern_p = 1 - hps.sv_rate if prefix != "test" else 1.0
                sv_mask = (
                    torch.rand(encod_data.shape, generator=sv_gen) < bern_p
                ).float()
            else:
                # Create a placeholder tensor
                sv_mask = torch.ones(n_samps, 0, 0)
            # Load or simulate external inputs
            if f"{prefix}_ext_input" in data_dict:
                ext_input = to_tensor(data_dict[f"{prefix}_ext_input"])
            else:
                ext_input = torch.zeros(n_samps, n_steps, 0)
            if f"{prefix}_truth" in data_dict:
                # Load or simulate ground truth TODO: use None instead of NaN?
                cf = data_dict["conversion_factor"]
                truth = to_tensor(data_dict[f"{prefix}_truth"]) / cf
            else:
                truth = torch.full((n_samps, 0, 0), float("nan"))
            # Remove unnecessary data during IC encoder segment
            sv_mask = sv_mask[:, hps.dm_ic_enc_seq_len :]
            ext_input = ext_input[:, hps.dm_ic_enc_seq_len :]
            truth = truth[:, hps.dm_ic_enc_seq_len :, :]
            # Extract data for any extra keys
            other = [to_tensor(data_dict[f"{prefix}_{k}"]) for k in extra_keys]
            return (
                SessionBatch(
                    encod_data=encod_data,
                    recon_data=recon_data,
                    ext_input=ext_input,
                    truth=truth,
                    sv_mask=sv_mask,
                ),
                tuple(other),
            )

        # Store the data for each session
        all_train_data.append(create_session_batch("train", extra_keys))
        all_valid_data.append(create_session_batch("valid", extra_keys))
        if "test_encod_data" in data_dict:
            all_test_data.append(create_session_batch("test"))
    # Store the datasets on the datamodule
    datamodule.train_data = all_train_data
    datamodule.train_ds = [SessionDataset(*train_data) for train_data in all_train_data]
    datamodule.valid_data = all_valid_data
    datamodule.valid_ds = [SessionDataset(*valid_data) for valid_data in all_valid_data]
    if len(all_test_data) == len(all_train_data):
        datamodule.test_data = all_test_data
        datamodule.test_ds = [SessionDataset(*test_data) for test_data in all_test_data]


def reshuffle_train_valid(data_dict, seed, ratio=None):
    # Identify the data to be reshuffled
    data_keys = [k.replace("train_", "") for k in data_dict.keys() if "train_" in k]
    # Combine all training and validation data arrays
    arrays = [
        np.concatenate([data_dict["train_" + k], data_dict["valid_" + k]])
        for k in data_keys
    ]
    # Reshuffle and split training and validation data
    valid_size = ratio if ratio is not None else len(data_dict["valid_" + data_keys[0]])
    arrays = train_test_split(*arrays, test_size=valid_size, random_state=seed)
    train_arrays = [a for i, a in enumerate(arrays) if (i - 1) % 2]
    valid_arrays = [a for i, a in enumerate(arrays) if i % 2]
    # Replace the previous data with the newly split data
    for k, ta, va in zip(data_keys, train_arrays, valid_arrays):
        data_dict.update({"train_" + k: ta, "valid_" + k: va})
    return data_dict

class SessionDataset(Dataset):
    def __init__(
        self, model_tensors: SessionBatch, extra_tensors: tuple
    ):
        all_tensors = [*model_tensors, *extra_tensors]
        assert all(
            all_tensors[0].size(0) == tensor.size(0) for tensor in all_tensors
        ), "Size mismatch between tensors"
        self.model_tensors = model_tensors
        self.extra_tensors = extra_tensors

    def __getitem__(self, index):
        model_tensors = SessionBatch(*[t[index] for t in self.model_tensors])
        extra_tensors = tuple(t[index] for t in self.extra_tensors)
        return model_tensors, extra_tensors

    def __len__(self):
        return len(self.model_tensors[0])

class SessionAreaDataset(Dataset):
    def __init__(
        self,
        data_dict: dict,
        info_dict: dict,
    ):
        self.data_list = list(data_dict.values()) # keys: batch index, value: {``area_name``: tensor of shape (time, num neurons)}
        self.info_list = list(info_dict.values()) # keys: batch index, value: info string
        self.empty_dict = {key: np.zeros(list(value.shape[:-1]) + [0]) for key, value in self.data_list[0].items()}
        
    def __getitem__(self, idx):
        return AreaSessionBatch(
                    encod_data=self.data_list[idx],
                    recon_data=self.data_list[idx],
                    info_data=self.info_list[idx],
                    ext_input=self.empty_dict,
                    sv_mask=self.empty_dict, # implement sv_mask elsewhere
                )
    
    def __len__(self): return len(self.data_list)

class BasicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_paths: List[str],
        batch_keys: List[str] = [],
        attr_keys: List[str] = [],
        batch_size: int = 64,
        reshuffle_tv_seed: int = None,
        reshuffle_tv_ratio: float = None,
        sv_rate: float = 0.0,
        sv_seed: int = 0,
        dm_ic_enc_seq_len: int = 0,
    ):
        assert (
            reshuffle_tv_seed is None or len(attr_keys) == 0
        ), "Dataset reshuffling is incompatible with the `attr_keys` argument."
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        hps = self.hparams
        data_dicts = []
        for data_path in hps.data_paths:
            # Load data arrays from the file
            with h5py.File(data_path, "r") as h5file:
                data_dict = {k: v[()] for k, v in h5file.items()}
            # Reshuffle the training / validation split
            if hps.reshuffle_tv_seed is not None:
                data_dict = reshuffle_train_valid(
                    data_dict, hps.reshuffle_tv_seed, hps.reshuffle_tv_ratio
                )
            data_dicts.append(data_dict)
        # Attach data to the datamodule
        attach_tensors(self, data_dicts, extra_keys=hps.batch_keys)
        for attr_key in hps.attr_keys:
            setattr(self, attr_key, data_dict[attr_key])

    def train_dataloader(self, shuffle=True):
        # PTL provides all batches at once, so divide amongst dataloaders
        batch_size = int(self.hparams.batch_size / len(self.train_ds))
        dataloaders = {
            i: DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
            )
            for i, ds in enumerate(self.train_ds)
        }
        return CombinedLoader(dataloaders, mode="max_size_cycle")

    def val_dataloader(self):
        # PTL provides all batches at once, so divide amongst dataloaders
        batch_size = int(self.hparams.batch_size / len(self.valid_ds))
        dataloaders = {
            i: DataLoader(
                ds,
                batch_size=batch_size,
            )
            for i, ds in enumerate(self.valid_ds)
        }
        return CombinedLoader(dataloaders, mode="max_size_cycle")

    def predict_dataloader(self):
        # NOTE: Returning dicts of DataLoaders is incompatible with trainer.predict,
        # but convenient for posterior sampling. Can't use CombinedLoader here because
        # we only want to see each sample once.
        dataloaders = {
            s: {
                "train": DataLoader(
                    self.train_ds[s],
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                ),
                "valid": DataLoader(
                    self.valid_ds[s],
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                ),
            }
            for s in range(len(self.train_ds))
        }
        # Add the test dataset if it is available
        if hasattr(self, "test_ds"):
            for s in range(len(self.train_ds)):
                dataloaders[s]["test"] = DataLoader(
                    self.test_ds[s],
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                )
        return dataloaders

class MesoMapDataModule(pl.LightningDataModule):
    """
    Mesoscale activity map data from 
        ``Brain-wide neural activity underlying memory-guided movement``, Chen et al. (2023).
    
    Reads a .h5 file with the hierarchical structure of:
        file (subject) --> group (session) --> dataset (area data or info)
        
        - ``file`` is named as "sub-{subject_id}"
        - ``session`` is named as "ses-{session_id}"
        - ``area data`` is named as "area-{abbreviation}", and has attribute ``type`` == "data"
        - ``info`` is named as "{information_type}", and has attribute ``type`` == "info"
    """
    def __init__(
        self,
        subject_id: int,
        session_idx: Union[int, str],
        area_names: list,
        batch_size: int = 16,
        p_split: list = [0.85, 0.15],
    ):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        hps = self.hparams
        to_string = lambda l: [s.decode('utf-8') for s in l]
        
        filename = f'{PATHS.datapath}sub-{hps.subject_id}/sub-{hps.subject_id}.h5'
        with h5py.File(filename, 'r') as file:

            # Use sessions according to session_idx
            session_names = list(file.keys())
            if hps.session_idx == "all": session_idxs = list(range(len(session_names)))
            else: session_idxs = [hps.session_idx]
            
            # Only retain sessions according to available keys
            dataset_names = [f"area-{key}" for key in hps.area_names]
            session_idxs = [si for si in session_idxs if
                           np.all([dsname in file[session_names[si]].keys() for dsname in dataset_names])]

            self.train_session_datasets = []
            self.val_session_datasets = []
            for si in session_idxs:
                group = file[session_names[si]]
                
                # Filter data by photostim_onset
                filter1 = group["photostim_onset"][:]
                included_batches = np.where(filter1 == b"N/A")[0]
                batch_dim = len(included_batches)
                
                # Get information necessary
                info1 = group["trial_instruction"][:][included_batches]
                info2 = group["outcome"][:][included_batches]
                info_strings = ["_".join(to_string(info)) for info in zip(info1, info2)]

                # Turn data into dictionary, then SessionAreaDataset
                area_data_dict = {}
                area_info_dict = {}
                for dataset_name in dataset_names:
                    area_name = dataset_name.replace("area-", "")
                    ds = group[dataset_name]
                    assert ds.attrs.get("type") == "data"

                    arr = ds[:][included_batches]
                    arr = np.swapaxes(arr, 1, 2)
                    for bi in range(batch_dim):
                        if dataset_name == dataset_names[0]: area_data_dict[bi] = {}
                        area_data_dict[bi][area_name] = arr[bi]
                        area_info_dict[bi] = info_strings[bi]
                session_dataset = SessionAreaDataset(area_data_dict, area_info_dict)
                train_ds, val_ds = random_split(session_dataset, hps.p_split)
                self.train_session_datasets.append(train_ds)
                self.val_session_datasets.append(val_ds)
                
    def train_dataloader(self, shuffle=True):
        dataloaders = {
            i: DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                shuffle=shuffle,
                drop_last=False,
            )
            for i, ds in enumerate(self.train_session_datasets)
        }
        return CombinedLoader(dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        dataloaders = {
            i: DataLoader(
                ds,
                batch_size=len(ds),
                shuffle=False,
            )
            for i, ds in enumerate(self.val_session_datasets)
        }
        return CombinedLoader(dataloaders, mode="max_size_cycle")
