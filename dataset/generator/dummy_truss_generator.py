import numpy as np
import re
from typing import Dict

from Flamant.dataset.structural.analysis import LinearAnalysis
from Flamant.dataset.structural.structure import DummyTruss
from Flamant.dataset.generator.abstract_truss_generator import AbstractTrussGenerator


class DummyTrussGenerator(AbstractTrussGenerator):

    def __init__(self, config: Dict[str, int | float] | str | None = None, analysis=None, structure=None):
        super().__init__(config, analysis=analysis)
        if structure is None:
            self._structure: DummyTruss = DummyTruss()
        else:
            self._structure: DummyTruss = structure

    @property
    def default_config(self) -> Dict[str, Dict[str, str | int | float]]:
        config = {
            '__area__': {'distribution': 'constant', 'value': 1.e-2},
            '__young__': {'distribution': 'constant', 'value': 200.e9},
            'load': {'distribution': 'uniform', 'low': 0., 'high': -1000e3}
        }

        config.update({f"A_{i}": {'shared_with': '__area__'} for i in range(29)})
        config.update({f"E_{i}": {'shared_with': '__young__'} for i in range(29)})

        return config

    @property
    def structure(self) -> DummyTruss:
        return self._structure

    @property
    def analysis(self):
        return self._analysis

    def construct_result(self, params: Dict[str, float | int]) -> Dict[str, float]:
        keys = params.keys()

        keys_a = sorted(
            [s for s in keys if re.match(r"A_\d+", s)],
            key=lambda s: (s[:2], int(s[2:]))
        )
        keys_e = sorted(
            [s for s in keys if re.match(r"E_\d+", s)],
            key=lambda s: (s[:2], int(s[2:]))
        )

        r = {
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
