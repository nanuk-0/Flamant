import numpy as np
import re
from typing import Dict

from Flamant.dataset.structural.analysis import LinearAnalysis
from Flamant.dataset.structural.structure import PrattTruss
from Flamant.dataset.generator.abstract_truss_generator import AbstractTrussGenerator


class PrattTrussGenerator(AbstractTrussGenerator):
    """
    Generator class for 2D Pratt truss structures.

    This generator creates datasets for structural simulations of Pratt trusses
    with configurable panel count, geometry, and material properties. It uses a
    linear static analysis and outputs bar forces, displacements, and stiffness.

    Parameters
    ----------
    config : dict or str or None, optional
        YAML file path or parameter dictionary. If None, the default configuration is used.
    analysis : AbstractAnalysis, optional
        Analysis method to use. Defaults to `LinearAnalysis`.

    Attributes
    ----------
    structure : PrattTruss
        Structural model class for generating node and element topology.
    analysis : LinearAnalysis
        Linear static solver used to compute structural response.
    default_config : dict
        Dictionary of default parameter distributions and values.
    """

    def __init__(self, config: Dict[str, int | float] | str | None = None, analysis=None, bisupported=False, structure=None):
        """
        Initialize the PrattTrussGenerator with a given configuration and analysis.

        Parameters
        ----------
        config : dict or str or None
            Parameter configuration or YAML path.
        analysis : AbstractAnalysis, optional
            Structural analysis object.
        """
        super().__init__(config, analysis=analysis)
        if structure is None:
            self._structure = PrattTruss(bisupported=bisupported)
        else:
            self._structure = structure

    @property
    def default_config(self) -> Dict[str, Dict[str, str | int | float]]:
        """
        Returns the default configuration for parameter sampling.

        Returns
        -------
        dict
            Dictionary defining default parameter values and their distributions.

        Notes
        -----
        - 29 bars (A_0 to A_28, E_0 to E_28)
        - 16 nodes (P_x_0 to P_y_15)
        """
        config = {
            '__area__': {'distribution': 'constant', 'value': 1.e-2},
            '__young__': {'distribution': 'constant', 'value': 200.e9},
            'n_panels': {'distribution': 'constant_int', 'value': 8},
            'length': {'distribution': 'constant', 'value': 60.0},
            'height': {'distribution': 'constant', 'value': 7.5},
            'volumetric_weight': {'distribution': 'constant', 'value': 78.5e3},
        }

        config.update({f"A_{i}": {'shared_with': '__area__'} for i in range(29)})
        config.update({f"E_{i}": {'shared_with': '__young__'} for i in range(29)})
        config.update({f"P_x_{i}": {'distribution': 'constant', 'value': 0.} for i in range(16)})
        config.update({f"P_y_{i}": {'distribution': 'constant', 'value': 0.} for i in range(16)})

        return config

    @property
    def structure(self) -> PrattTruss:
        """
        Returns the truss structure used for generation.

        Returns
        -------
        PrattTruss
            Instance of the Pratt truss structure.
        """
        return self._structure

    @property
    def analysis(self) -> LinearAnalysis:
        """
        Returns the structural analysis method.

        Returns
        -------
        LinearAnalysis
            The analysis used for evaluating the truss.
        """
        return self._analysis

    def construct_result(self, params: Dict[str, float | int]) -> Dict[str, float]:
        """
        Generate result dictionary for one sample using the current structure and analysis.

        Parameters
        ----------
        params : dict
            Dictionary of input parameters used to generate the structure.

        Returns
        -------
        dict
            Dictionary containing simulation results and relevant input metadata:

            - 'length' : float
                Total span of the truss.
            - 'height' : float
                Total height of the truss.
            - 'n_panels' : int
                Number of vertical panels in the Pratt truss.
            - 'volumetric_weight' : float
                Density used to compute distributed loads.
            - 'nodes_coordinate' : np.ndarray
                Flattened array of node coordinates.
            - 'nodes_displacement' : np.ndarray
                Flattened array of nodal displacements.
            - 'nodes_load' : np.ndarray
                Flattened nodal load vector from OpenSees.
            - 'bars_area' : np.ndarray
                Cross-sectional areas for each bar.
            - 'bars_young' : np.ndarray
                Young’s modulus values for each bar.
            - 'bars_force' : np.ndarray
                Internal bar forces.
            - 'bars_length_init' : np.ndarray
                Initial lengths of each bar.
            - 'bars_elongation' : np.ndarray
                Elongation of each bar (before - after).
            - 'bars_strain' : np.ndarray
                Normal strain (elongation / initial length).
            - 'stiffness_matrix' : np.ndarray
                Flattened global stiffness matrix.
            - 'connectivity_matrix' : np.ndarray
                Flattened element connectivity array (pairs of node indices).
        """
        keys = params.keys()

        keys_a = sorted(
            [s for s in keys if re.match(r"A_\d+", s)],
            key=lambda s: (s[:2], int(s[2:]))
        )
        keys_e = sorted(
            [s for s in keys if re.match(r"E_\d+", s)],
            key=lambda s: (s[:2], int(s[2:]))
        )
        keys_p = sorted(
            [s for s in keys if re.match(r"P_[x,y]_\d+", s)],
            key=lambda s: (s[:4], int(s[4:]))
        )
        keys_p = tuple(zip(keys_p[:len(keys_p) // 2], keys_p[len(keys_p) // 2:]))

        r = {
            'length': params['length'],
            'height': params['height'],
            'n_panels': params['n_panels'],
            'volumetric_weight': params['volumetric_weight'],
            'nodes_coordinate': self.structure.nodes_coordinates.reshape(-1),
            'nodes_displacement': self.structure.nodes_displacements.reshape(-1),
            'nodes_load': np.array(self.structure.loads).reshape(-1),
            'nodes_external_load': np.array(self.structure.external_load).reshape(-1),
            'bars_area': np.array([params[k] for k in keys_a]),
            'bars_young': np.array([params[k] for k in keys_e]),
            'bars_force': self.structure.elements_forces.reshape(-1),
            'bars_length_init': self.structure.initial_elements_length,
            'bars_strain': self.structure.bars_strain,
            'stiffness_matrix': self.structure.stiffness_matrix.reshape(-1),
            'connectivity_matrix': self.structure.elements_connectivity.reshape(-1),
            'support_reactions': self.structure.supports_reactions.reshape(-1),
        }

        return r
