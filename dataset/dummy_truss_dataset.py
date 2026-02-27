from typing import List, Callable

import numpy as np
import torch

from Flamant.dataset import AbstractHDF5Dataset
import h5py


class DummyTrussDataset(AbstractHDF5Dataset):
    def __init__(self,
                 filepath: str,
                 f_noise_length: Callable[[tuple], np.ndarray] | None = None,
                 f_noise_loads: Callable[[tuple], np.ndarray] | None = None,
                 f_noise_strain: Callable[[tuple], np.ndarray] | None = None,
                 f_noise_displacement: Callable[[tuple], np.ndarray] | None = None,
                 dtype=torch.float32, bisupported=False):
        super().__init__(filepath)
        self.bisupported = bisupported

        # Noise configuration
        self.f_noise_length = f_noise_length or (lambda shape: np.ones(shape))
        self.f_noise_loads = f_noise_loads or (lambda shape: np.ones(shape))
        self.f_noise_displacement = f_noise_displacement or (lambda shape: np.ones(shape))
        self.f_noise_strain = f_noise_strain or (lambda shape: np.ones(shape))

        # Database extraction
        self.dtype = dtype
        with h5py.File(filepath, 'r') as f:
            self.nodes_coordinate = np.vstack(f['nodes_coordinate'][:], dtype=np.float64)
            self.nodes_displacement = np.vstack(f['nodes_displacement'][:], dtype=np.float64)
            self.load = np.vstack(f['nodes_load'][:], dtype=np.float64)
            self.external_load = np.vstack(f['nodes_external_load'][:], dtype=np.float64)
            self.bars_area = np.vstack(f['bars_area'][:], dtype=np.float64)
            self.bars_young = np.vstack(f['bars_young'][:], dtype=np.float64)
            self.bars_force = np.vstack(f['bars_force'][:], dtype=np.float64)
            self.bars_length_init = np.vstack(f['bars_length_init'][:], dtype=np.float64)
            self.bars_strain = np.vstack(f['bars_strain'][:], dtype=np.float64)
            self.stiffness_matrix = np.vstack(f['stiffness_matrix'][:], dtype=np.float64)
            self.connectivity_matrix = np.vstack(f['connectivity_matrix'][:])
            self.support_reaction = np.vstack(f['support_reactions'][:])

        self.n_nodes = self.nodes_coordinate.shape[1] // 2
        self.n_elems = self.bars_area.shape[1]

        self.self_weight_load = self.load.copy()
        self.self_weight_load[:, [3]] -= self.external_load

        self.connectivity_matrix = self.connectivity_matrix.reshape((-1, self.n_elems, 2))

        # Noise application
        self.noise_bars_length_init = self.f_noise_length(self.bars_length_init.shape)
        self.noise_nodes_displacement = self.f_noise_displacement(self.nodes_displacement.shape)
        self.noise_load = self.f_noise_loads(self.external_load.shape)
        noise = self.f_noise_strain(self.bars_force.shape)
        self.noise_bars_force = noise
        self.noise_bars_strain = noise

    def __getitems__(self, idx: List[int]):
        n_nodes = len(self.nodes_coordinate[0]) // 2

        data_1 = self.nodes_displacement[idx] * self.noise_nodes_displacement[idx]
        data_1 = data_1[:, [2, 3, 4, 6, 7]]
        data_2 = self.external_load[idx] * self.noise_load[idx]
        data_3 = self.bars_strain[idx] * self.noise_bars_strain[idx]

        data = np.hstack([data_1, data_2, data_3])

        data = torch.tensor(data, dtype=self.dtype)
        target = torch.tensor(self.bars_area[idx] * self.bars_young[idx], dtype=self.dtype)
        load = torch.tensor(self.load[idx].reshape((-1, 2 * n_nodes, 1)), dtype=self.dtype)
        displacements = torch.tensor(self.nodes_displacement[idx].reshape((-1, 2 * self.n_nodes, 1)), dtype=self.dtype)
        self_weight = torch.tensor(self.self_weight_load[idx].reshape((-1, self.n_nodes, 2)), dtype=self.dtype)
        support_reactions = torch.tensor(self.support_reaction[idx].reshape((-1, self.n_nodes, 2)), dtype=self.dtype)

        # x, y, u, q, w, r
        return [[data[i], target[i], displacements[i], load[i], self_weight[i], support_reactions[i]]
                for i in range(len(idx))]

    def __len__(self):
        return len(self.nodes_coordinate)