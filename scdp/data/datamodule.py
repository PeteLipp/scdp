import json
import random
from typing import Optional, Dict
from pathlib import Path
from tqdm import tqdm

import numpy as np
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset, random_split
from lightning.pytorch.utilities import CombinedLoader

from scdp.common.pyg import DataLoader
from mldft.ml.data.components.loader import OFLoader
from scdp.data.dataloader import ProbeDataLoader
from scdp.data.md_dataset import SmallDensityDataset

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.datamodule import RandomSubsetPerEpoch


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)

class DataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        dataset: DictConfig,
        split_file: Optional[Path],
        num_workers: DictConfig,
        batch_size: DictConfig,
        move_n_samples_from_train_to_val: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        with open(split_file, "r") as fp:
            self.splits = json.load(fp)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.metadata: Optional[Dict] = None
        self.move_n_samples_from_train_to_val = move_n_samples_from_train_to_val
        
    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        if self.move_n_samples_from_train_to_val > 0:
            train_split = self.splits["train"]
            val_split = self.splits["validation"]
            # move n random samples from train to val:
            additional_val_samples = random.sample(
                train_split, self.move_n_samples_from_train_to_val
            )
            self.splits["validation"] = val_split + additional_val_samples
            self.splits["train"] = [
                i for i in train_split if i not in additional_val_samples
            ]

        self.train_dataset = Subset(self.dataset, self.splits["train"])
        self.val_dataset = Subset(self.dataset, self.splits["validation"])
        self.test_dataset = Subset(self.dataset, self.splits["test"])
        if (Path(self.dataset.path) / 'metadata.json').exists():
            with open(Path(self.dataset.path) / 'metadata.json', 'r') as fp:
                self.metadata = json.load(fp)
        else:
            self.metadata = self.get_metadata()

    def get_metadata(self):
        x_sum = 0
        x_2 = 0
        unique_atom_types = set()
        avg_num_neighbors = 0
        print('get metadata.')
        progress = tqdm(total=len(self.train_dataset))
        for data in self.train_dataset:
            x_sum += data.chg_labels.mean()
            x_2 += (data.chg_labels ** 2).mean()
            unique_atom_types.update(data.atom_types.numpy().tolist())
            avg_num_neighbors += data.edge_index.shape[1] / len(data.atom_types) / 2
            progress.update(1)
        x_mean = x_sum / len(self.train_dataset)
        x_var = x_2 / len(self.train_dataset) - x_mean ** 2
        avg_num_neighbors = int(avg_num_neighbors / len(self.train_dataset))
        # this is the avg num neighbors without the probes
        metadata = {
            'target_mean': x_mean.item(), 
            'target_var': x_var.item(), 
            'avg_num_neighbors': avg_num_neighbors,
            'unique_atom_types': list(unique_atom_types)
        }
        with open(self.dataset.path / 'metadata.json', 'w') as fp:
            json.dump(metadata, fp)
        return metadata

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
        )
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.dataset=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


class ProbeDataModule(DataModule):
    def __init__(
        self,
        dataset: DictConfig,
        split_file: Optional[Path],
        num_workers: DictConfig,
        batch_size: DictConfig,
        n_probe: DictConfig,
        basis_info: BasisInfo,
        do_dengen_val: bool = False,
        n_dengen_mols: int | None = None,
        move_n_samples_from_train_to_val: int = 0,
        collator_kwargs: Optional[Dict] = None,
    ):
        super().__init__(dataset=dataset, split_file=split_file, num_workers=num_workers, batch_size=batch_size,
                            move_n_samples_from_train_to_val=move_n_samples_from_train_to_val)
        self.n_probe = n_probe
        self.basis_info = basis_info
        self.do_dengen_val = do_dengen_val
        self.n_dengen_mols = n_dengen_mols
        self.collator_kwargs = collator_kwargs if collator_kwargs is not None else {}

    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        super().setup(stage)
        if self.do_dengen_val:
            # Dengen uses base 'labels' dir
            if self.n_dengen_mols is not None and len(self.val_dataset) >= self.n_dengen_mols:
                # dengen_val_subset = val_data[-self.n_dengen_mols :]
                self.dengen_val_subset = Subset(
                    self.val_dataset, 
                    range(len(self.val_dataset) - self.n_dengen_mols, len(self.val_dataset))
                )

                self.val_sampler = RandomSubsetPerEpoch(
                    dataset_size=len(self.val_dataset),
                    subset_size=self.n_dengen_mols,
                    base_seed=None,  # Use a fixed seed for reproducibility
                )

            else:
                self.dengen_val_subset = self.val_dataset
                self.val_sampler = None

    def train_dataloader(self, shuffle=True):
        return ProbeDataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            n_probe=self.n_probe.train,
            worker_init_fn=worker_init_fn,
            basis_info=self.basis_info,
            collator_kwargs=self.collator_kwargs,
        )

    def val_dataloader(self):
        val_loader = ProbeDataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            n_probe=self.n_probe.val,
            worker_init_fn=worker_init_fn,
            basis_info=self.basis_info,
            collator_kwargs=self.collator_kwargs,
        )

        if self.do_dengen_val:
            dengen_val_loader = ProbeDataLoader(
                self.dengen_val_subset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                n_probe=self.n_probe.val,
                worker_init_fn=worker_init_fn,
                basis_info=self.basis_info,
                collator_kwargs=self.collator_kwargs,
            )
            dengen_val_loader_random = ProbeDataLoader(
                self.val_dataset,
                sampler=self.val_sampler,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                n_probe=self.n_probe.val,
                worker_init_fn=worker_init_fn,
                basis_info=self.basis_info,
                collator_kwargs=self.collator_kwargs,
            )
            combined_loader = CombinedLoader(
                {
                    "val": val_loader,
                    "dengen_val": dengen_val_loader,
                    "dengen_val_random": dengen_val_loader_random,
                },
                mode="sequential",
            )

            return combined_loader
        else:
            return val_loader
    
    def test_dataloader(self):
        return ProbeDataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
            n_probe=self.n_probe.test,
            worker_init_fn=worker_init_fn,
            basis_info=self.basis_info,
            collator_kwargs=self.collator_kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.dataset=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
            f"{self.n_probe=})"
        )
    


class SmallProbeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        mol_name: str,
        num_workers: DictConfig,
        batch_size: DictConfig,
        basis_info: BasisInfo,
        use_n_train_mols_for_val: int = 0,
        do_dengen_val: bool = False,
        n_dengen_mols: int | None = None,
        dataset_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.root = root
        self.mol_name = mol_name
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.basis_info = basis_info
        self.dataset_kwargs = dataset_kwargs if dataset_kwargs is not None else {}
        
        self.do_dengen_val = do_dengen_val
        self.n_dengen_mols = n_dengen_mols
        self.use_n_train_mols_for_val = use_n_train_mols_for_val
        if self.use_n_train_mols_for_val == 0:
            print("No validation set available for dengen.")
            self.do_dengen_val = False
            self.n_dengen_mols = None

    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        # initialize datasets
        self.train_dataset = SmallDensityDataset(
            root=self.root,
            mol_name=self.mol_name,
            split="train",
            basis_info=self.basis_info,
            **self.dataset_kwargs)
        
        self.test_dataset = SmallDensityDataset(
            root=self.root,
            mol_name=self.mol_name,
            split="test",
            basis_info=self.basis_info,
            **self.dataset_kwargs)
        
        # create random validation set from train set:
        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset, 
            [len(self.train_dataset) - self.use_n_train_mols_for_val, self.use_n_train_mols_for_val],
            generator=torch.Generator().manual_seed(42)
        )

        if self.do_dengen_val and self.use_n_train_mols_for_val > 0:
            # Dengen uses base 'labels' dir
            if self.n_dengen_mols is not None and len(self.val_dataset) >= self.n_dengen_mols:
                # dengen_val_subset = val_data[-self.n_dengen_mols :]
                self.dengen_val_subset = Subset(
                    self.val_dataset, 
                    range(len(self.val_dataset) - self.n_dengen_mols, len(self.val_dataset))
                )

                self.val_sampler = RandomSubsetPerEpoch(
                    dataset_size=len(self.val_dataset),
                    subset_size=self.n_dengen_mols,
                    base_seed=None,  # Use a fixed seed for reproducibility
                )

            else:
                self.dengen_val_subset = self.val_dataset
                self.val_sampler = None

    def train_dataloader(self, shuffle=True):
        return OFLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        val_loader = OFLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )

        if self.do_dengen_val:
            dengen_val_loader = OFLoader(
                self.dengen_val_subset,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn,
            )
            dengen_val_loader_random = OFLoader(
                self.val_dataset,
                sampler=self.val_sampler,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                worker_init_fn=worker_init_fn,
            )
            combined_loader = CombinedLoader(
                {
                    "val": val_loader,
                    "dengen_val": dengen_val_loader,
                    "dengen_val_random": dengen_val_loader_random,
                },
                mode="sequential",
            )

            return combined_loader
        else:
            return val_loader
    
    def test_dataloader(self):
        return OFLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.dataset=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )