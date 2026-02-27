from Flamant.dataset.structural.structure.abstract_planar_truss import *


class PrattTruss(AbstractPlanarTruss):
    """
    Planar Pratt truss structure generator.

    This class implements the `generate_structure` method for building a Pratt truss model
    with arbitrary panel count and dimensions. It includes node creation, bar elements
    (horizontal, vertical, diagonal), support conditions, and distributed load handling.

    In a Pratt truss:
    - Top and bottom chords are horizontal.
    - Verticals connect the top and bottom nodes.
    - Diagonal bars alternate to resist tension/compression in a symmetric layout.

    Parameters
    ----------
    params : dict
        A dictionary containing geometric, material, and loading parameters:

        Required keys:
        - "length" : float
            Total horizontal span of the truss.
        - "height" : float
            Total vertical height of the truss.
        - "n_panels" : int
            Number of vertical panels. Must be even.
        - "volumetric_weight" : float
            Material density used to compute distributed loads.
        - "E_{i}" : float
            Young’s modulus for the i-th element.
        - "A_{i}" : float
            Cross-sectional area for the i-th element.
        - "P_x_{i}" : float
            Horizontal point load at node i.
        - "P_y_{i}" : float
            Vertical point load at node i.
    """

    def __init__(self, bisupported=False):
        super().__init__()
        self.bisupported = bisupported
        self._external_loads = None

    def generate_structure(self, params: Dict[str, int | float]) -> None:
        """
        Generate the full Pratt truss structure in OpenSees.

        Includes creation of nodes, material assignment, truss elements (top, bottom,
        vertical, diagonal), application of distributed loads based on geometry and
        area, and fixed supports at both ends.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters (see class docstring for required keys).

        Raises
        ------
        Exception
            If `n_panels` is not even.
        """
        self.params = params.copy()
        length = float(params['length'])
        height = float(params["height"])
        n_panels = int(params["n_panels"])
        volumetric_weight = float(params["volumetric_weight"])
        panel_width = length / n_panels
        if n_panels % 2:
            raise Exception("n_panels must be pair")

        n_nodes = 2 * n_panels
        self._external_loads = np.zeros((n_nodes, self.n_dof))
        self._self_weight_loads = np.zeros((n_nodes, self.n_dof))

        # Parse initial load
        for i in range(n_nodes):
            p_x = self.params[f"P_x_{i}"]
            p_y = self.params[f"P_y_{i}"]
            if p_x != 0 or p_y != 0:
                self._external_loads[i, 0] = p_x
                self._external_loads[i, 1] = p_y

        # Nodes
        for idx in range(n_panels + 1):
            x = idx * panel_width
            ops.node(idx, x, 0.)
        for idx in range(n_panels + 1, 2 * n_panels):
            x = length - panel_width * (idx - n_panels)
            ops.node(idx, x, height)

        idx_bar = 0

        # Horizontal top chords
        for idx in range(n_panels):
            area, young = self._get_bar_characteristics(idx_bar)
            idx_start, idx_end = idx, idx + 1
            ops.uniaxialMaterial('Elastic', idx_bar, young)
            ops.element('Truss', idx_bar, idx_start, idx_end, area, idx_bar)
            self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
            idx_bar += 1

        # Horizontal bottom chords
        for idx in range(n_panels + 1, 2 * n_panels - 1):
            area, young = self._get_bar_characteristics(idx_bar)
            idx_start, idx_end = idx, idx + 1
            ops.uniaxialMaterial('Elastic', idx_bar, young)
            ops.element('Truss', idx_bar, idx_start, idx_end, area, idx_bar)
            self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
            idx_bar += 1

        # Vertical elements
        for idx in range(1, n_panels):
            area, young = self._get_bar_characteristics(idx_bar)
            idx_start = idx
            idx_end = 2 * n_panels - idx_start
            ops.uniaxialMaterial('Elastic', idx_bar, young)
            ops.element('Truss', idx_bar, idx_start, idx_end, area, idx_bar)
            self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
            idx_bar += 1

        # Diagonal: first bar
        area, young = self._get_bar_characteristics(idx_bar)
        ops.uniaxialMaterial('Elastic', idx_bar, young)
        ops.element('Truss', idx_bar, 0, 2 * n_panels - 1, area, idx_bar)
        self._set_bar_load(0, 2 * n_panels - 1, area, volumetric_weight)
        idx_bar += 1

        # Diagonal (left side)
        for idx in range(1, n_panels // 2):
            area, young = self._get_bar_characteristics(idx_bar)
            idx_start = idx + 1
            idx_end = 2 * n_panels + 1 - idx_start
            ops.uniaxialMaterial('Elastic', idx_bar, young)
            ops.element('Truss', idx_bar, idx_start, idx_end, area, idx_bar)
            self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
            idx_bar += 1

        # Diagonal (right side)
        for idx in range(n_panels // 2, n_panels - 1):
            area, young = self._get_bar_characteristics(idx_bar)
            idx_start = idx
            idx_end = 2 * n_panels - 1 - idx_start
            ops.uniaxialMaterial('Elastic', idx_bar, young)
            ops.element('Truss', idx_bar, idx_start, idx_end, area, idx_bar)
            self._set_bar_load(idx_start, idx_end, area, volumetric_weight)
            idx_bar += 1

        # Diagonal (middle)
        area, young = self._get_bar_characteristics(idx_bar)
        ops.uniaxialMaterial('Elastic', idx_bar, young)
        ops.element('Truss', idx_bar, n_panels, n_panels + 1, area, idx_bar)
        self._set_bar_load(n_panels, n_panels + 1, area, volumetric_weight)
        idx_bar += 1

        # Supports
        ops.fix(0, 1, 1)
        if self.bisupported:
            ops.fix(n_panels, 1, 1)
        else:
            ops.fix(n_panels, 0, 1)

        self.params[f"P_x_0"] = 0.
        self.params[f"P_y_0"] = 0.
        self.params[f"P_y_{n_panels}"] = 0.

        # Loads
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        for idx, sx, sy in self.supports:
            if sx:
                self._external_loads[idx, 0] = 0.
                self._self_weight_loads[idx, 0] = 0.
            if sy:
                self._external_loads[idx, 1] = 0.
                self._self_weight_loads[idx, 1] = 0.

        for idx, load in enumerate(self._self_weight_loads + self._external_loads):
            ops.load(idx, *load)

    def _get_bar_characteristics(self, idx):
        """
        Retrieve the area and Young’s modulus for a given element.

        Parameters
        ----------
        idx : int
            The index of the bar.

        Returns
        -------
        area : float
            Cross-sectional area of the bar.
        young : float
            Young’s modulus of the bar.
        """
        young = self.params[f"E_{idx}"]
        area = self.params[f"A_{idx}"]
        return area, young

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
        #self.params[f"P_y_{idx_start}"] -= 0.5 * q
        #self.params[f"P_y_{idx_end}"] -= 0.5 * q

    @property
    def external_load(self):
        return self._external_loads

    @property
    def self_weight_load(self):
        return self._self_weight_loads