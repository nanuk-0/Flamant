from Flamant.dataset.structural.structure.abstract_planar_truss import *


class DummyTruss(AbstractPlanarTruss):
    def __init__(self):
        super().__init__()
        self._external_loads = None

    def generate_structure(self, params: Dict[str, int | float]) -> None:
        self.params = params.copy()

        self._external_loads = params['load']
        self._self_weight_loads = np.zeros((4, self.n_dof))

        volumetric_weight = float(params["volumetric_weight"])

        # Nodes
        ops.node(0, 0., 0.)
        ops.node(1, 5., 0.)
        ops.node(2, 10., 0.)
        ops.node(3, 5., 5.)

        # Members
        connectivity = [[0, 1], [1, 2], [2, 3], [3, 0], [3, 1]]
        for i, (j, k) in enumerate(connectivity):
            young = self.params[f"E_{i}"]
            area = self.params[f"A_{i}"]

            ops.uniaxialMaterial('Elastic', i, young)
            ops.element('Truss', i, j, k, area, i)

            self._set_bar_load(j, k, area, volumetric_weight)

        # Supports
        ops.fix(0, 1, 1)
        ops.fix(2, 0, 1)

        # Loads
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        self._self_weight_loads[0, 0] = 0.
        self._self_weight_loads[0, 1] = 0.
        self._self_weight_loads[2, 1] = 0.

        for i, (qx, qy) in enumerate(self._self_weight_loads):
            if i == 1: qy += self._external_loads
            ops.load(i, qx, qy)

    def _set_bar_load(self, idx_start, idx_end, area, density):
        """
        Applies distributed load from a bar to its connected nodes.

        The load is computed based on the bar's volume and density, then
        split equally between the start and end nodes as point loads.

        Parameters
        ----------
        idx_start : int
            Index of the start node.
        idx_end : int
            Index of the end node.
        area : float
            Cross-sectional area of the bar.
        density : float
            Volumetric weight (density) of the material.
        """
        c_start = self.nodes_coordinates[idx_start, :]
        c_end = self.nodes_coordinates[idx_end, :]
        length = np.linalg.norm(c_start - c_end)
        volume = length * area
        q = volume * density

        self._self_weight_loads[idx_start, 1] -= 0.5 * q
        self._self_weight_loads[idx_end, 1] -= 0.5 * q

    @property
    def external_load(self):
        return self._external_loads

    @property
    def self_weight_load(self):
        return self._self_weight_loads
