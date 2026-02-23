"""HTP50cork — HeatSolver (발췌/요약)

주의:
- 이 파일은 app.py의 HeatSolver 구현에서 핵심 로직만 발췌하여 보고서 부록용으로 정리한 것입니다.
- 벡터화/버퍼 최적화/GUI 관련 코드는 생략되어 있습니다.
- 전체 원본은 code/app.py를 참고하세요.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix


class HeatSolver:
    """1D transient heat transfer (+ optional pyrolysis) solver.

    State vector:
        y = [T_0 ... T_{N-1}, Tmax_0 ... Tmax_{N-1}]  (size = 2N)
    """

    def __init__(self, params, materials_data, bc_data, enable_pyrolysis=True):
        self.p = params
        self.mat_data = materials_data
        self.bc_data = bc_data
        self.enable_pyrolysis = enable_pyrolysis

        # Geometry (bilayer)
        self.L1 = params['L1']          # cork thickness
        self.L2 = params['L2']          # metal thickness
        self.N1 = int(params['N1'])     # cork nodes
        self.N2 = int(params['N2'])     # metal nodes
        self.dx1 = self.L1 / self.N1
        self.dx2 = self.L2 / self.N2
        self.N = self.N1 + self.N2
        self.m = self.N1               # interface index

        self._init_interpolators()
        self.jac_sparsity = self._build_jac_sparsity()
        self.t_current = 0.0

    def _build_jac_sparsity(self):
        """Provide sparsity pattern for the BDF Jacobian approximation."""
        N = self.N
        size = 2 * N
        J = lil_matrix((size, size), dtype=int)

        # Temperature equations: tri-diagonal coupling
        for i in range(N):
            J[i, i] = 1
            if i > 0:     J[i, i-1] = 1
            if i < N-1:   J[i, i+1] = 1
            # pyrolysis affects cork-side properties via Tmax
            if i <= self.m:
                J[i, N+i] = 1

        # Tmax equations: depends on T and Tmax
        for i in range(N):
            row = N + i
            J[row, i] = 1
            if i > 0:     J[row, i-1] = 1
            if i < N-1:   J[row, i+1] = 1
            J[row, N+i] = 1

        return J.tocsr()

    def _init_interpolators(self):
        """Create property interpolators k(T), Cp(T), rho(T), and BC h(t), Tr(t)."""

        def create_interp(data_list):
            arr = np.array(data_list, dtype=float)
            x = arr[:, 0]
            f_k   = interp1d(x, arr[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
            f_Cp  = interp1d(x, arr[:, 2], kind='linear', bounds_error=False, fill_value="extrapolate")
            f_rho = interp1d(x, arr[:, 3], kind='linear', bounds_error=False, fill_value="extrapolate")
            return f_k, f_Cp, f_rho

        # Cork and Metal property tables
        self.k1_func, self.Cp1_func, self.rho1_func = create_interp(self.mat_data['cork'])
        self.k2_func, self.Cp2_func, self.rho2_func = create_interp(self.mat_data['metal'])

        # Normalized mass profile M(Tmax) for Advanced mode
        mass = np.array(self.mat_data.get('mass', []), dtype=float)
        if len(mass) > 0:
            self.mass_func = interp1d(mass[:, 0], mass[:, 1], kind='linear',
                                      bounds_error=False, fill_value=(mass[0, 1], mass[-1, 1]))
        else:
            self.mass_func = lambda x: 100.0

        # Boundary conditions
        htab  = np.array(self.bc_data['h'], dtype=float)
        Trtab = np.array(self.bc_data['Tr'], dtype=float)
        self.h_func  = interp1d(htab[:, 0],  htab[:, 1],  kind='linear',
                               bounds_error=False, fill_value=(htab[0, 1],  htab[-1, 1]))
        self.Tr_func = interp1d(Trtab[:, 0], Trtab[:, 1], kind='linear',
                               bounds_error=False, fill_value=(Trtab[0, 1], Trtab[-1, 1]))

    @staticmethod
    def k_harm(k1, k2):
        """Harmonic mean conductivity."""
        return (2.0 * k1 * k2) / (k1 + k2 + 1e-12)

    @staticmethod
    def sigmoid(x, stiffness=100.0):
        """Smooth gate function to avoid discontinuities."""
        limit = 700.0 / (stiffness + 1e-12)
        x = np.clip(x, -limit, limit)
        return 1.0 / (1.0 + np.exp(-stiffness * x))

    def get_props1(self, T, Tmax):
        """Cork properties with optional pyrolysis effects."""
        k   = np.maximum(self.k1_func(T),   1e-3)
        Cp  = np.maximum(self.Cp1_func(T),  1.0)
        rho = np.maximum(self.rho1_func(T), 1e-20)

        if not self.enable_pyrolysis:
            return k, np.maximum(Cp, 500.0), rho

        Tcrit = float(self.p.get('T_simple', 500.0))
        mode  = str(self.p.get('pyro_mode', 'advanced')).lower()

        # Freeze Cp/k after Tmax exceeds Tcrit AND node is cooling.
        g_thr  = self.sigmoid(Tmax - Tcrit, stiffness=20.0)
        g_cool = self.sigmoid((Tmax - 0.1) - T, stiffness=20.0)
        factor = g_thr * g_cool

        kTmax  = np.maximum(self.k1_func(Tmax), 1e-3)
        CpTmax = np.maximum(self.Cp1_func(Tmax), 1.0)

        k  = k  * (1.0 - factor) + kTmax  * factor
        Cp = Cp * (1.0 - factor) + CpTmax * factor

        # Advanced: density from mass profile (fixed-grid assumption)
        if mode != 'simple':
            mass_pct = np.clip(self.mass_func(Tmax), 20.0, 100.0)  # [%]
            rho = rho * (mass_pct / 100.0)

        return k, Cp, np.maximum(rho, 1e-20)

    def get_props2(self, T):
        """Metal properties."""
        k   = np.maximum(self.k2_func(T),   1e-3)
        Cp  = np.maximum(self.Cp2_func(T),  500.0)
        rho = np.maximum(self.rho2_func(T), 1e-20)
        return k, Cp, rho

    def ode_system(self, t, y):
        """Compute RHS of coupled ODE system (T and Tmax)."""
        self.t_current = t
        N = self.N
        T    = y[:N]
        Tmax = y[N:]

        dTdt = np.zeros(N)

        # Boundary values
        h  = max(0.0, float(self.h_func(t)))
        Tr = float(self.Tr_func(t))

        # Properties
        k1, Cp1, rho1 = self.get_props1(T[:self.m],   Tmax[:self.m])
        k2, Cp2, rho2 = self.get_props2(T[self.m:])

        # ---- Heat equation (schematic) ----
        # 0) Surface: convection + conduction to node 1
        k01 = self.k_harm(k1[0], k1[1])
        q_cond = k01 * (T[0] - T[1]) / self.dx1
        q_conv = h * (Tr - T[0])
        dTdt[0] = (q_conv - q_cond) / (rho1[0] * Cp1[0] * (self.dx1 / 2))

        # 1) Interior nodes (vectorized in full code; omitted here)
        # 2) Interface: mixed heat capacity of half-cells (cork + metal)
        # 3) Metal interior and bottom adiabatic boundary
        # ------------------------------------------------------------

        # ---- History equation: Tmax grows only when T is rising ----
        ramp = np.maximum(0.0, dTdt)
        gate = self.sigmoid(T - Tmax, stiffness=100.0)
        dTmax_dt = ramp * gate

        return np.concatenate([dTdt, dTmax_dt])

    def solve(self):
        """Time integration using SciPy BDF."""
        t_span = (0.0, float(self.p['t_final']))

        T0    = np.ones(self.N) * float(self.p['T_init'])
        Tmax0 = np.ones(self.N) * float(self.p['T_init'])
        y0 = np.concatenate([T0, Tmax0])

        sol = solve_ivp(
            self.ode_system, t_span, y0,
            method='BDF',
            rtol=1e-6, atol=1e-8,
            first_step=1e-4,
            jac_sparsity=self.jac_sparsity,
        )
        return sol
