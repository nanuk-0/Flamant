from typing import List, Callable

import numpy as np
import torch

from Flamant.dataset import AbstractHDF5Dataset
import h5py


class FixedPrattTrussDataset(AbstractHDF5Dataset):
    """
    Dataset class for Fixed Pratt Truss simulations with optional input noise injection.

    This dataset provides structured access to nodal and element-based truss data,
    supporting the injection of multiplicative noise into features for robustness
    or data augmentation.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file containing the truss simulation data.
    f_noise_length : Callable[[tuple], np.ndarray], optional
        Function that generates multiplicative noise for truss lengths and bar lengths.
        The function should accept a shape/size tuple and return an array of matching shape.
        Defaults to a function that returns ones (i.e., no noise).
    f_noise_loads : Callable[[tuple], np.ndarray], optional
        Function that generates multiplicative noise for node loads.
        The function should accept a shape/size tuple and return an array of matching shape.
        Defaults to a function that returns ones.
    f_noise_strain : Callable[[tuple], np.ndarray], optional
        Function that generates multiplicative noise for strain, elongation, and force.
        The function should accept a shape/size tuple and return an array of matching shape.
        Defaults to a function that returns ones.
    f_noise_displacement : Callable[[tuple], np.ndarray], optional
        Function that generates multiplicative noise for nodal displacements.
        The function should accept a shape/size tuple and return an array of matching shape.
        Defaults to a function that returns ones.
    dtype : torch.dtype, optional
        Data type for the returned tensors. Defaults to `torch.float32`.

    Attributes
    ----------
    height : np.ndarray
        Truss height values.
    length : np.ndarray
        Truss span/length values.
    n_panels : np.ndarray
        Number of panels in each truss.
    nodes_coordinate : np.ndarray
        2D coordinates of the nodes.
    nodes_displacement : np.ndarray
        Displacement values at each node.
    load : np.ndarray
        Load vectors applied to each node.
    bars_area : np.ndarray
        Cross-sectional area of the bars.
    bars_young : np.ndarray
        Young’s modulus values for each bar.
    bars_force : np.ndarray
        Internal bar forces.
    bars_length_init : np.ndarray
        Initial bar lengths before deformation.
    bars_elongation : np.ndarray
        Elongation values of the bars.
    bars_strain : np.ndarray
        Strain values of the bars.
    stiffness_matrix : np.ndarray
        Global stiffness matrix of the truss system.
    noise_* : np.ndarray
        Precomputed noise arrays for corresponding features.
    """

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
            self.height = f['height'][:].astype(np.float64)
            self.length = f['length'][:].astype(np.float64)
            self.n_panels = f['n_panels'][:].astype(np.int64)
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

        self.self_weight_load = self.load - self.external_load
        self.connectivity_matrix = self.connectivity_matrix.reshape((-1, self.n_elems, 2))

        # Noise application
        self.noise_length = self.f_noise_length(self.height.shape)
        self.noise_truss_width = self.f_noise_length(self.length.shape)
        self.noise_bars_length_init = self.f_noise_length(self.bars_length_init.shape)
        self.noise_nodes_displacement = self.f_noise_displacement(self.nodes_displacement.shape)
        self.noise_load = self.f_noise_loads(self.load.shape)
        noise = self.f_noise_strain(self.bars_force.shape)
        self.noise_bars_force = noise
        self.noise_bars_strain = noise

    def __getitems__(self, idx: List[int]):
        """
        Retrieves a list of training samples corresponding to the provided indices.

        Parameters
        ----------
        idx : List[int]
            Indices of the samples to retrieve.

        Returns
        -------
        List[List[torch.Tensor]]
            Each element is a list of 5 tensors:
            - data : torch.Tensor
                Input features, including selected displacements, loads, and strains.
            - target : torch.Tensor
                Element-wise product of bar area and Young's modulus.
            - nodes : torch.Tensor
                Node coordinates reshaped to (n_nodes, 2).
            - displacements : torch.Tensor
                Node displacements reshaped to (2 * n_nodes, 1).
            - load : torch.Tensor
                Load vectors reshaped to (2 * n_nodes, 1).
        """
        n_nodes = len(self.nodes_coordinate[0]) // 2

        data_1 = self.nodes_displacement[idx] * self.noise_nodes_displacement[idx]
        if self.bisupported:  # Remove displacement on horizontal dof !
            data_1 = data_1[:, [k for k in range(4 * self.n_panels[0])
                                if k not in (0, 1, 2 * self.n_panels[0], 2 * self.n_panels[0] + 1)]]
        else:
            data_1 = data_1[:, [k for k in range(4 * self.n_panels[0]) if k not in (0, 1, 2 * self.n_panels[0] + 1)]]
        data_2 = self.external_load[idx] * self.noise_load[idx]
        data_2 = data_2[:, [i for i in range(3, self.n_panels[0] * 2, 2)]]
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
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            The total number of truss samples.
        """
        return len(self.height)


class FixedPrattTrussDatasetThreeTargets(FixedPrattTrussDataset):
    """
    Variant of `FixedPrattTrussDataset` returning only three target values per sample.

    The three values are from bars 0, 14, 22.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    f_noise_length : Callable[[tuple], np.ndarray], optional
        Noise function for length-related inputs. Accepts the shape as argument.
    f_noise_loads : Callable[[tuple], np.ndarray], optional
        Noise function for node loads. Accepts the shape as argument.
    f_noise_strain : Callable[[tuple], np.ndarray], optional
        Noise function for strain and force data. Accepts the shape as argument.
    f_noise_displacement : Callable[[tuple], np.ndarray], optional
        Noise function for node displacements. Accepts the shape as argument.
    dtype : torch.dtype, optional
        Tensor dtype.
    """

    def __init__(self,
                 filepath: str,
                 f_noise_length: Callable[[tuple], np.ndarray] | None = None,
                 f_noise_loads: Callable[[tuple], np.ndarray] | None = None,
                 f_noise_strain: Callable[[tuple], np.ndarray] | None = None,
                 f_noise_displacement: Callable[[tuple], np.ndarray] | None = None,
                 dtype=torch.float32, bisupported=False):
        super().__init__(filepath=filepath,
                         f_noise_length=f_noise_length,
                         f_noise_loads=f_noise_loads,
                         f_noise_strain=f_noise_strain,
                         f_noise_displacement=f_noise_displacement,
                         dtype=dtype, bisupported=bisupported)
        self.bars_area = self.bars_area[:, [0, 14, 22]]
        self.bars_young = self.bars_young[:, [0, 14, 22]]


class FixedPrattTrussDatasetSingleTarget(FixedPrattTrussDataset):
    """
    Variant of `FixedPrattTrussDataset` returning only one target value per sample.
    The reference value is assumed to be the one of the first bar.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    f_noise_length : Callable[[tuple], np.ndarray], optional
        Noise function for length-related inputs. Accepts the shape as argument.
    f_noise_loads : Callable[[tuple], np.ndarray], optional
        Noise function for node loads. Accepts the shape as argument.
    f_noise_strain : Callable[[tuple], np.ndarray], optional
        Noise function for strain and force data. Accepts the shape as argument.
    f_noise_displacement : Callable[[tuple], np.ndarray], optional
        Noise function for node displacements. Accepts the shape as argument.
    dtype : torch.dtype, optional
        Tensor dtype.
    """

    def __init__(self,
                 filepath: str,
                 f_noise_length: Callable[[tuple], np.ndarray] | None = None,
                 f_noise_loads: Callable[[tuple], np.ndarray] | None = None,
                 f_noise_strain: Callable[[tuple], np.ndarray] | None = None,
                 f_noise_displacement: Callable[[tuple], np.ndarray] | None = None,
                 dtype=torch.float32, bisupported=False):
        super().__init__(filepath=filepath,
                         f_noise_length=f_noise_length,
                         f_noise_loads=f_noise_loads,
                         f_noise_strain=f_noise_strain,
                         f_noise_displacement=f_noise_displacement,
                         dtype=dtype, bisupported=bisupported)
        self.bars_area = self.bars_area[:, 0:1]
        self.bars_young = self.bars_young[:, 0:1]
