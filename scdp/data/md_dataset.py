"""
Taken from https://github.com/ccr-cheng/InfGCN-pytorch/blob/main/datasets/small_density.py
and modified to fit into the SCDP framework.
"""


import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import Data

from mldft.ofdft.basis_integrals import get_normalization_vector
from mldft.utils.molecules import build_molecule_np
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.data.components.convert_transforms import ToTorch
from scdp.data.vnode import get_virtual_nodes
from scdp.common.constants import bohr2ang


ATOM_TYPES = {
    'benzene': torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
    'ethanol': torch.LongTensor([0, 0, 2, 1, 1, 1, 1, 1, 1]),
    'phenol': torch.LongTensor([0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1]),
    'resorcinol': torch.LongTensor([0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1]),
    'ethane': torch.LongTensor([0, 0, 1, 1, 1, 1, 1, 1]),
    'malonaldehyde': torch.LongTensor([2, 0, 0, 0, 2, 1, 1, 1, 1]),
}
TYPE_TO_Z = {0: 6, 1: 1, 2: 8}


class SmallDensityDataset(Dataset):
    def __init__(self, root, mol_name, split, basis_info, use_vnodes=False, n_probe=None, transforms=None):
        """
        Density dataset for small molecules in the MD datasets.
        Note that the validation and test splits are the same.
        :param root: data root
        :param mol_name: name of the molecule
        :param split: data split, can be 'train', 'validation', 'test'
        """
        super(SmallDensityDataset, self).__init__()
        assert mol_name in ('benzene', 'ethanol', 'phenol', 'resorcinol', 'ethane', 'malonaldehyde')
        self.root = root
        self.mol_name = mol_name
        self.split = split
        if split == 'validation':
            split = 'test'

        self.n_grid = 50  # number of grid points along each dimension
        self.grid_size = 20.  # box size in Bohr
        self.data_path = os.path.join(root, mol_name, f'{mol_name}_{split}')

        self.atom_type = ATOM_TYPES[mol_name]
        self.atom_charges = torch.tensor([TYPE_TO_Z[i.item()] for i in self.atom_type], dtype=torch.long)
        self.atom_coords = np.load(os.path.join(self.data_path, 'structures.npy'))
        self.densities = self._convert_fft(np.load(os.path.join(self.data_path, 'dft_densities.npy')))
        self.grid_coord = self._generate_grid()

        self.basis_info = basis_info
        self.use_vnodes = use_vnodes
        self.n_probe = n_probe
        self.transforms = transforms

    def _convert_fft(self, fft_coeff):
        # The raw data are stored in Fourier basis, we need to convert them back.
        print(f'Precomputing {self.split} density from FFT coefficients ...')
        fft_coeff = torch.FloatTensor(fft_coeff).to(torch.complex64)
        d = fft_coeff.view(-1, self.n_grid, self.n_grid, self.n_grid)
        hf = self.n_grid // 2
        # first dimension
        d[:, :hf] = (d[:, :hf] - d[:, hf:] * 1j) / 2
        d[:, hf:] = torch.flip(d[:, 1:hf + 1], [1]).conj()
        d = torch.fft.ifft(d, dim=1)
        # second dimension
        d[:, :, :hf] = (d[:, :, :hf] - d[:, :, hf:] * 1j) / 2
        d[:, :, hf:] = torch.flip(d[:, :, 1:hf + 1], [2]).conj()
        d = torch.fft.ifft(d, dim=2)
        # third dimension
        d[..., :hf] = (d[..., :hf] - d[..., hf:] * 1j) / 2
        d[..., hf:] = torch.flip(d[..., 1:hf + 1], [3]).conj()
        d = torch.fft.ifft(d, dim=3)
        return torch.flip(d.real.view(-1, self.n_grid ** 3), [-1]).detach()

    def _generate_grid(self):
        x = torch.linspace(self.grid_size / self.n_grid, self.grid_size, self.n_grid)
        return torch.stack(torch.meshgrid(x, x, x, indexing='ij'), dim=-1).view(-1, 3).detach()
    

    def subsample_grid(self, of_data, n_probe):
        assert isinstance(n_probe, int), "n_probe must be an integer"
        if n_probe < of_data.n_probe:
            indices = torch.randperm(of_data.n_probe)[:n_probe]
            of_data.probe_coords = of_data.probe_coords[indices]
            of_data.chg_labels = of_data.chg_labels[indices]
            of_data.n_probe[:] = n_probe
        return of_data


    def __getitem__(self, item):

        if self.use_vnodes:
            # convert to coords to angstrom for vnode generation:
            atom_coords = self.atom_coords[item].clone()
            atom_coords = atom_coords * bohr2ang

            virtual_nodes = get_virtual_nodes(
                atom_coords=atom_coords, 
                pbc=False, 
                method='bond',
                atom_types=self.atom_charges,
                struct=None,  # only used for pbc
            )

            # convert back to bohr:
            atom_coords = atom_coords / bohr2ang
            virtual_nodes = virtual_nodes / bohr2ang

            # use oxygen (8) as the atomic number for vnodes:
            charges = torch.cat([self.atom_charges, 8 * torch.ones(len(virtual_nodes), dtype=torch.long)], dim=0)
            coords = np.concatenate([atom_coords, virtual_nodes], axis=0)
            is_vnode = torch.cat([torch.zeros(len(self.atom_charges), dtype=torch.bool), 
                                torch.ones(len(virtual_nodes), dtype=torch.bool)])
        else:
            charges = self.atom_charges
            coords = self.atom_coords[item]
            is_vnode = torch.zeros(len(self.atom_charges), dtype=torch.bool)

        mol = build_molecule_np(charges=charges.numpy(),
                        positions=coords.astype(np.float64), basis=self.basis_info.basis_dict)
        
        # reset vnode atomic numbers:
        charges[is_vnode] = 0

        of_data = OFData.construct_new(
            basis_info=self.basis_info,
            pos=mol.atom_coords(),
            atomic_numbers=charges.numpy(),
            coeffs=np.zeros(mol.nao),
            add_irreps=self.basis_info is not None,
            dual_basis_integrals=get_normalization_vector(mol),
        )
        of_data = ToTorch()(of_data)

        of_data.add_item("n_probe", torch.tensor([self.grid_coord.shape[0]]), Representation.NONE)
        of_data.add_item("grid_size", (torch.ones(3) * self.n_grid).view(1, 3), Representation.VECTOR)
        of_data.add_item("cell", (torch.eye(3) * self.grid_size).view(1, 3, 3), Representation.VECTOR)
        of_data.add_item("probe_coords", self.grid_coord, Representation.VECTOR)
        of_data.add_item("chg_labels", self.densities[item], Representation.SCALAR)
        of_data.add_item("molwise_n_atom", torch.tensor([len(self.atom_charges)], dtype=int), Representation.NONE)
        of_data.add_item("ground_state_coeffs", torch.zeros_like(of_data.coeffs), Representation.VECTOR)
        of_data.add_item("molwise_n_vnode", torch.tensor([int(is_vnode.sum().item())], dtype=int), Representation.NONE)
        of_data.add_item("is_vnode", is_vnode, Representation.SCALAR)

        if self.n_probe is not None:
            of_data = self.subsample_grid(of_data, self.n_probe)

        if self.transforms is not None:
            of_data = self.transforms(of_data)

        return of_data

    def __len__(self):
        return self.atom_coords.shape[0]
