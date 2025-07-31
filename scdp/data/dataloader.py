
import torch
import numpy as np
from typing import Optional, List
from torch.utils.data import DataLoader

from scdp.common.pyg import Collater
from mldft.utils.molecules import build_molecule_np
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.data.components.convert_transforms import ToTorch, AddFullEdgeIndex, AddRadiusEdgeIndex, AddLocalFramesSAD
from mldft.ml.data.components.basis_transforms import AddLocalFrames
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics


class ProbeCollater(Collater):
    def __init__(self, follow_batch, exclude_keys, n_probe: int = None, basis_info: BasisInfo = None, 
                 add_lframes: bool = False, edge_radial_cutoff: float = None, vnode_dict: dict = None,
                 remove_vnodes: bool = False, lframes_sad_statistics: DatasetStatistics = None):
        super().__init__(follow_batch, exclude_keys)
        self.n_probe = n_probe
        self.basis_info = basis_info
        if add_lframes:
            self.add_lframes_module = AddLocalFrames()
        else:
            self.add_lframes_module = None
        if edge_radial_cutoff is None:
            self.add_edge_index_module = AddFullEdgeIndex()
        else:
            self.add_edge_index_module = AddRadiusEdgeIndex(radius=self.edge_radial_cutoff)

        self.vnode_dict = {} if vnode_dict is None else vnode_dict
        self.remove_vnodes = remove_vnodes
        if lframes_sad_statistics is not None:
            self.lframes_sad_module = AddLocalFramesSAD(
                dataset_statistics=lframes_sad_statistics,
                basis_info=self.basis_info,
            )
        else:
            self.lframes_sad_module = None

    def remove_vnodes_from_sample(self, x):
        """Remove vnodes from the sample."""
        x.atom_types = x.atom_types[~x.is_vnode]
        x.coords = x.coords[~x.is_vnode]
        x.n_vnode = 0
        
        # at last change the is_vnode attribute:
        x.is_vnode = x.is_vnode[~x.is_vnode]
        return x

    def __call__(self, batch):
        of_data_list = []
        for x in batch:
            if self.remove_vnodes:
                x = self.remove_vnodes_from_sample(x)
            for vnode_number, atomic_number in self.vnode_dict.items():
                x.atom_types[x.atom_types == vnode_number] = atomic_number
            mol = build_molecule_np(charges = x.atom_types.numpy(),
                        positions = x.coords.numpy().astype(np.float64), basis = self.basis_info.basis_dict)
            of_data = ToTorch()(OFData.minimal_sample_from_mol(mol, self.basis_info))
            of_data = self.add_edge_index_module(of_data)
            
            # add random noise to virtual nodes to break collinearity
            # TODO: replace this with a more sophisticated method
            of_data.pos[x.is_vnode] += torch.randn_like(of_data.pos[x.is_vnode]) * 1e-4

            if self.add_lframes_module is not None:
                of_data = self.add_lframes_module(of_data)

            if self.lframes_sad_module is not None:
                of_data = self.lframes_sad_module(of_data)

            if self.n_probe is not None:
                assert isinstance(self.n_probe, int), "n_probe must be an integer"
                if self.n_probe < x.n_probe:
                    x = x.sample_probe(n_probe=self.n_probe)


            # add relevant attributes from x to of_data and drop the rest:
            of_data.add_item("n_probe", x.n_probe, Representation.NONE)
            of_data.add_item("grid_size", x.grid_size, Representation.NONE)
            of_data.add_item("cell", x.cell, Representation.VECTOR)  # TODO: or dual vector?
            of_data.add_item("probe_coords", x.probe_coords, Representation.VECTOR)
            of_data.add_item("chg_labels", x.chg_labels, Representation.SCALAR)
            of_data.add_item("molwise_n_vnode", x.n_vnode, Representation.NONE)
            of_data.add_item("molwise_n_atom", x.n_atom, Representation.NONE)
            of_data.add_item("is_vnode", x.is_vnode, Representation.SCALAR)
            of_data.add_item("ground_state_coeffs", torch.zeros_like(of_data.coeffs), Representation.VECTOR)
            of_data_list.append(of_data)

        of_batch = OFBatch.from_data_list(of_data_list, follow_batch=["coeffs", "atomic_numbers"])
        return of_batch


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
        n_probe: int = None,
        follow_batch: Optional[List[str]] = [None],
        exclude_keys: Optional[List[str]] = [None],
        basis_info: BasisInfo = None,
        collator_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.n_probe = n_probe
        self.basis_info = basis_info
        collator_kwargs = {} if collator_kwargs is None else collator_kwargs

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=ProbeCollater(follow_batch, exclude_keys, n_probe,
                                     basis_info=basis_info,
                                     **collator_kwargs),
            **kwargs,
        )