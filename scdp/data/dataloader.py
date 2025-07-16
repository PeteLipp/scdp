
from typing import Optional, List
from torch.utils.data import DataLoader

from scdp.common.pyg import Collater
from mldft.utils.molecules import build_molecule_np
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.data.components.convert_transforms import ToTorch
from mldft.ml.data.components.basis_info import BasisInfo
import numpy as np


class ProbeCollater(Collater):
    def __init__(self, follow_batch, exclude_keys, n_probe=200, basis_info: BasisInfo = None):
        super().__init__(follow_batch, exclude_keys)
        self.n_probe = n_probe
        self.basis_info = basis_info

    def __call__(self, batch):
        of_data_list = []
        basis_info = self.basis_info #BasisInfo.from_atomic_numbers_with_even_tempered_basis(basis='def2-qzvppd', atomic_numbers=[1, 6, 7, 8, 9], even_tempered=False, beta=2.5, uncontracted=True)
        for x in batch:
            mol = build_molecule_np(charges = x.atom_types.numpy(),
                        positions = x.coords.numpy().astype(np.float64), basis = basis_info.basis_dict)
            of_data = ToTorch()(OFData.minimal_sample_from_mol(mol, basis_info))
            of_data.add_item("molecule", mol, Representation.NONE)
            of_data_list.append(of_data)

        if self.n_probe < x.n_probe:
            batch = [x.sample_probe(n_probe=min(self.n_probe, x.n_probe)) for x in batch]
        batch = super().__call__(batch)
        of_batch = OFBatch.from_data_list(of_data_list, ["coeffs", "atomic_numbers"])
        batch.of_batch = of_batch
        return batch


class ProbeDataLoader(DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        n_probe: int = 200,
        follow_batch: Optional[List[str]] = [None],
        exclude_keys: Optional[List[str]] = [None],
        basis_info: BasisInfo = None,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.n_probe = n_probe
        self.basis_info = basis_info

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=ProbeCollater(follow_batch, exclude_keys, n_probe,
                                     basis_info=basis_info),
            **kwargs,
        )