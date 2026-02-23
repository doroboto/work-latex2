import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
from tkinter import filedialog
from tkinter.colorchooser import askcolor
import csv
import time
import copy
import io
import threading
import os
import json
from default_data import (
    DEFAULT_CORK_CSV,
    DEFAULT_CORK_NO_PYRO_CSV,
    DEFAULT_P50_MASS_CSV,
    DEFAULT_METAL_CSV,
    DEFAULT_H_CSV,
    DEFAULT_TR_CSV
)

# UI style
UI_FONT_TITLE = ("Segoe UI", 17, "bold")
UI_FONT_SECTION = ("Segoe UI", 12, "bold")
UI_FONT_LABEL = ("Segoe UI", 11)
UI_FONT_BUTTON = ("Segoe UI", 11, "bold")

COLOR_WARN = "#FFD60A"                      # Apple system yellow
COLOR_SECTION_TEXT = "#F2F2F7"               # Apple primary text
COLOR_LABEL_TEXT = "#98989D"                 # Apple secondary label
COLOR_BTN_SOFT_BLUE = "#0A84FF"              # Apple accent blue
COLOR_BTN_SOFT_BLUE_HOVER = "#3399FF"
COLOR_BTN_SOFT_GREEN = "#30D158"             # Apple system green
COLOR_BTN_SOFT_GREEN_HOVER = "#2AB84C"
COLOR_BTN_SOFT_RED = "#FF453A"               # Apple system red
COLOR_BTN_SOFT_RED_HOVER = "#E03E35"

try:
    import winsound as _winsound
except ImportError:
    _winsound = None


def _play_notify(kind="info"):
    """Play a pleasant notification chime."""
    if _winsound is None:
        return
    try:
        if kind == "error":
            _winsound.Beep(523, 100)
            _winsound.Beep(440, 150)
        elif kind == "warning":
            _winsound.Beep(659, 150)
        else:
            _winsound.Beep(784, 80)
            _winsound.Beep(1047, 120)
    except Exception:
        pass


def _show_msg(title, message, kind="info"):
    """Custom styled message dialog with pleasant notification sound."""
    _play_notify(kind)
    dialog = ctk.CTkToplevel()
    dialog.title(title)
    dialog.resizable(False, False)
    dialog.attributes("-topmost", True)
    dialog.focus_force()

    frame = ctk.CTkFrame(dialog, fg_color="transparent")
    frame.pack(fill="both", expand=True, padx=24, pady=(20, 16))

    ctk.CTkLabel(
        frame, text=message, font=UI_FONT_LABEL,
        wraplength=340, justify="center",
    ).pack(expand=True)

    ctk.CTkButton(
        frame, text="OK", width=80, command=dialog.destroy,
        fg_color="#0A84FF", hover_color="#3399FF", text_color="white",
        font=UI_FONT_BUTTON,
    ).pack(pady=(12, 0))

    dialog.update_idletasks()
    w = max(dialog.winfo_reqwidth() + 20, 320)
    h = max(dialog.winfo_reqheight() + 20, 140)
    sx = dialog.winfo_screenwidth()
    sy = dialog.winfo_screenheight()
    dialog.geometry(f"{w}x{h}+{(sx - w) // 2}+{(sy - h) // 2}")
    dialog.after(10, dialog.grab_set)
    dialog.wait_window()


DEFAULT_KEY_MAP = {
    'cork': 'DEFAULT_CORK_CSV',
    'cork_no_pyro': 'DEFAULT_CORK_NO_PYRO_CSV',
    'mass': 'DEFAULT_P50_MASS_CSV',
    'metal': 'DEFAULT_METAL_CSV',
    'h': 'DEFAULT_H_CSV',
    'Tr': 'DEFAULT_TR_CSV',
}

USER_DEFAULTS_VERSION_KEY = "__defaults_version"
USER_DEFAULTS_VERSION = 2
LINKED_CORE_KEYS = ("cork", "cork_no_pyro", "mass")
DEFAULT_HEADERS_BY_KEY = {
    "cork": ["Temp", "k", "Cp", "rho"],
    "cork_no_pyro": ["Temp", "k", "Cp", "rho"],
    "mass": ["Temp", "MassNorm"],
    "metal": ["Temp", "k", "Cp", "rho"],
    "h": ["Time", "Value"],
    "Tr": ["Time", "Value"],
}


def get_user_defaults_path():
    base_dir = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    app_dir = os.path.join(base_dir, "HTP50cork")
    os.makedirs(app_dir, exist_ok=True)
    return os.path.join(app_dir, "user_defaults.json")

# =============================================================================
# 1. Physics Engine (Mathematica-Flow Replica)
# =============================================================================
class HeatSolver:
    def __init__(self, params, materials_data, bc_data, enable_pyrolysis=True):
        self.p = params
        self.mat_data = materials_data
        self.bc_data = bc_data
        self.enable_pyrolysis = enable_pyrolysis
        
        # Geometry
        self.L1 = params['L1']
        self.L2 = params['L2']
        self.N1 = int(params['N1'])
        self.N2 = int(params['N2'])
        
        self.dx1 = self.L1 / self.N1
        self.dx2 = self.L2 / self.N2
        self.N = self.N1 + self.N2
        self.m = self.N1

        # Preallocate buffers to reduce per-call allocations in ode_system
        self._metal_len = max(0, self.N - self.m - 2)
        self._k_R_cork = np.empty(self.m - 1) if self.m > 1 else None
        self._k_L_metal = np.empty(self._metal_len) if self._metal_len > 0 else None
        self._k_R_metal = np.empty(self._metal_len) if self._metal_len > 0 else None
        
        # X coordinates
        x1 = np.linspace(self.dx1/2, self.L1 - self.dx1/2, self.N1)
        x2 = np.linspace(self.L1 + self.dx2/2, self.L1 + self.L2 - self.dx2/2, self.N2)
        self.x_grid = np.concatenate([x1, x2])
        
        self._init_interpolators()
        self.jac_sparsity = self._build_jac_sparsity()
        self.t_current = 0.0

    def _build_jac_sparsity(self):
        # Sparse structure for BDF Jacobian approximation
        N = self.N
        size = 2 * N
        J = lil_matrix((size, size), dtype=int)

        # T equations: tri-diagonal in T, diagonal in Tmax for cork region
        for i in range(N):
            J[i, i] = 1
            if i > 0:
                J[i, i - 1] = 1
            if i < N - 1:
                J[i, i + 1] = 1
            if i <= self.m:
                J[i, N + i] = 1

        # Tmax equations: tri-diagonal in T, diagonal in Tmax
        for i in range(N):
            row = N + i
            J[row, i] = 1
            if i > 0:
                J[row, i - 1] = 1
            if i < N - 1:
                J[row, i + 1] = 1
            J[row, N + i] = 1

        return J.tocsr()

    def _init_interpolators(self):
        # [Math-Flow 1] Linear Extrapolation (Match Mathematica)
        # Mathematica: "ExtrapolationHandler" -> Automatic (Linear)
        
        def create_clamped_interp(data_list, default_val):
            if not data_list:
                return lambda x: np.full_like(x, default_val)
            
            arr = np.array(data_list, dtype=float)
            x = arr[:, 0]
            # interp1d creates a function
            f_k = interp1d(x, arr[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
            f_Cp = interp1d(x, arr[:, 2], kind='linear', bounds_error=False, fill_value="extrapolate")
            f_rho = interp1d(x, arr[:, 3], kind='linear', bounds_error=False, fill_value="extrapolate")
            return f_k, f_Cp, f_rho

        # Cork
        self.k1_func, self.Cp1_func, self.rho1_func = create_clamped_interp(self.mat_data.get('cork'), 0.0)
        # Metal
        self.k2_func, self.Cp2_func, self.rho2_func = create_clamped_interp(self.mat_data.get('metal'), 0.0)
        # Fig.1 normalized mass profile (Temp [degC], MassNorm [%])
        mass_data = self.mat_data.get('mass')
        if mass_data:
            d = np.array(mass_data, dtype=float)
            self.mass_func = interp1d(
                d[:, 0], d[:, 1], kind='linear', bounds_error=False, fill_value=(d[0, 1], d[-1, 1])
            )
            self.mass_char_pct = float(np.clip(np.min(d[:, 1]), 1e-6, 99.9))
        else:
            self.mass_func = lambda x: np.full_like(np.asarray(x, dtype=float), 100.0, dtype=float)
            self.mass_char_pct = 20.0

        # BCs
        if self.bc_data.get('h'):
            d = np.array(self.bc_data['h'])
            self.h_func = interp1d(d[:, 0], d[:, 1], kind='linear', bounds_error=False, fill_value=(d[0, 1], d[-1, 1]))
        else:
            raise ValueError("Boundary condition h(t) data is missing. Please load CSV.")

        if self.bc_data.get('Tr'):
            d = np.array(self.bc_data['Tr'])
            self.Tr_func = interp1d(d[:, 0], d[:, 1], kind='linear', bounds_error=False, fill_value=(d[0, 1], d[-1, 1]))
        else:
            raise ValueError("Boundary condition Tr(t) data is missing. Please load CSV.")

    def get_props1(self, T_current, T_max_state):
        # [Math-Flow 2] State-Dependent Properties
        # Property selection is based on T_max_state maintained by the solver.
        
        k = self.k1_func(T_current)
        Cp = self.Cp1_func(T_current)
        rho = self.rho1_func(T_current)
        
        # Stability Safeguards
        k = np.maximum(k, 0.001)
        rho = np.maximum(rho, 1e-20)
        Cp = np.maximum(Cp, 1.0)
        
        pyro_mode = str(self.p.get('pyro_mode', 'advanced')).lower()
        T_simple = float(self.p.get('T_simple', 500.0))

        if self.enable_pyrolysis:
            k_smooth = 20.0

            # --- Cp/k freeze (common to both Simple and Advanced) ---
            # Once Tmax exceeds T_critical and node is cooling, freeze Cp/k
            # to the value at Tmax (irreversible pyrolysis proxy).
            threshold_gate = self.sigmoid(T_max_state - T_simple, stiffness=k_smooth)
            cooling_gate = self.sigmoid((T_max_state - 0.1) - T_current, stiffness=k_smooth)
            factor = threshold_gate * cooling_gate

            k_target = self.k1_func(T_max_state)
            Cp_target = self.Cp1_func(T_max_state)
            k_target = np.maximum(k_target, 0.001)
            Cp_target = np.maximum(Cp_target, 1.0)
            Cp = Cp * (1.0 - factor) + Cp_target * factor
            k = k * (1.0 - factor) + k_target * factor
            Cp = np.maximum(Cp, 1.0)

            if pyro_mode != 'simple':
                # --- Advanced: density from mass profile (irreversible) ---
                # Fixed-grid assumption: volume unchanged, so mass loss = density loss.
                # rho(Tmax) = rho_virgin * mass_func(Tmax) / 100
                mass_pct = np.asarray(self.mass_func(T_max_state), dtype=float)
                mass_pct = np.clip(mass_pct, self.mass_char_pct, 100.0)
                rho = rho * (mass_pct / 100.0)
                rho = np.maximum(rho, 1e-20)
        else:
            Cp = np.maximum(Cp, 500.0)
            
        return k, Cp, rho

    def get_props2(self, T):
        k = self.k2_func(T)
        Cp = self.Cp2_func(T)
        rho = self.rho2_func(T)
        k = np.maximum(k, 0.001)
        rho = np.maximum(rho, 1e-20)
        Cp = np.maximum(Cp, 500.0)
        return k, Cp, rho

    def k_harm(self, k1, k2):
        return (2 * k1 * k2) / (k1 + k2 + 1e-12)

    def sigmoid(self, x, stiffness=100.0): # Match Mathematica stiffness
        # Mathematica's LogisticSigmoid equivalent
        # Clip the exponent argument directly (-700 ~ 700) to prevent overflow.
        # Prevent overflow in multiplication before clip
        limit = 700.0 / (stiffness + 1e-12)
        x_clipped = np.clip(x, -limit, limit)
        arg = -stiffness * x_clipped
        return 1.0 / (1.0 + np.exp(arg))

    def ode_system(self, t, y):
        # [Math-Flow 3] Coupled ODE System
        # y contains BOTH Temperature (T) and Max Temperature History (T_max)
        # y = [T_0, ..., T_N-1, T_max_0, ..., T_max_N-1]
        
        self.t_current = t
        N = self.N
        T = y[:N]
        T_max = y[N:]
        
        dTdt = np.zeros(N)
        
        # 1. Boundary Values
        h_val = max(0.0, float(self.h_func(t)))
        # In Celsius mode, Tr can be negative.
        Tr_val = float(self.Tr_func(t))

        # 2. Properties (Using State Variables T and T_max)
        # Cork
        T1_vec = T[:self.m]
        Tmax1_vec = T_max[:self.m]
        k1_vec, Cp1_vec, rho1_vec = self.get_props1(T1_vec, Tmax1_vec)
        
        # Interface Node (needs special care)
        Tm = T[self.m]
        Tmax_m = T_max[self.m]
        
        # Interface Properties (Cork side & Metal side)
        k1_m_arr, Cp1_m_arr, rho1_m_arr = self.get_props1(np.array([Tm]), np.array([Tmax_m]))
        k1_m, Cp1_m, rho1_m = k1_m_arr[0], Cp1_m_arr[0], rho1_m_arr[0]
        
        k2_m_arr, Cp2_m_arr, rho2_m_arr = self.get_props2(np.array([Tm]))
        k2_m, Cp2_m, rho2_m = k2_m_arr[0], Cp2_m_arr[0], rho2_m_arr[0]

        # Metal
        T2_vec = T[self.m+1:]
        k2_vec, Cp2_vec, rho2_vec = self.get_props2(T2_vec)
        
        # --- Heat Equation (dT/dt) ---
        
        # Node 0: Cork Surface Boundary
        k_eff_01 = self.k_harm(k1_vec[0], k1_vec[1])
        q_cond = k_eff_01 * (T[0] - T[1]) / self.dx1
        q_conv = h_val * (Tr_val - T[0])
        dTdt[0] = (q_conv - q_cond) / (rho1_vec[0] * Cp1_vec[0] * (self.dx1/2))
        
        # Nodes 1 to m-1 (Cork Interior)
        # [Optimized] Vectorized calculation
        if self.m > 1:
            k_L_vec = self.k_harm(k1_vec[:-1], k1_vec[1:])
            k_R_vec = self._k_R_cork
            if self.m > 2:
                k_R_vec[:-1] = self.k_harm(k1_vec[1:-1], k1_vec[2:])
            k_R_vec[-1] = self.k_harm(k1_vec[-1], k1_m)
            
            q_in_vec = k_L_vec * (T[0:self.m-1] - T[1:self.m]) / self.dx1
            q_out_vec = k_R_vec * (T[1:self.m] - T[2:self.m+1]) / self.dx1
            dTdt[1:self.m] = (q_in_vec - q_out_vec) / (rho1_vec[1:] * Cp1_vec[1:] * self.dx1)
            
        # Node m: Interface
        k_L_m = self.k_harm(k1_vec[-1], k1_m)
        k_R_m = self.k_harm(k2_m, k2_vec[0])
        
        q_in_m = k_L_m * (T[self.m-1] - T[self.m]) / self.dx1
        q_out_m = k_R_m * (T[self.m] - T[self.m+1]) / self.dx2
        
        heat_capacity_m = (rho1_m * Cp1_m * self.dx1/2) + (rho2_m * Cp2_m * self.dx2/2)
        dTdt[self.m] = (q_in_m - q_out_m) / heat_capacity_m
        
        # Nodes m+1 to N-2 (Metal Interior)
        # Metal region size
        metal_len = self._metal_len
        if metal_len > 0:
            k_L_vec = self._k_L_metal
            k_R_vec = self._k_R_metal

            # Left neighbors: k2_m then k2_vec[0:inner_len-1]
            k_L_vec[0] = self.k_harm(k2_m, k2_vec[0])
            if metal_len > 1:
                k_L_vec[1:] = self.k_harm(k2_vec[0:metal_len-1], k2_vec[1:metal_len])

            # Right neighbors: k2_vec[1:inner_len] then bottom node
            if metal_len > 1:
                k_R_vec[:-1] = self.k_harm(k2_vec[0:metal_len-1], k2_vec[1:metal_len])
            k_R_vec[-1] = self.k_harm(k2_vec[metal_len-1], k2_vec[metal_len])
            
            q_in_vec = k_L_vec * (T[self.m:self.N-2] - T[self.m+1:self.N-1]) / self.dx2
            q_out_vec = k_R_vec * (T[self.m+1:self.N-1] - T[self.m+2:self.N]) / self.dx2
            dTdt[self.m+1:self.N-1] = (q_in_vec - q_out_vec) / (rho2_vec[:-1] * Cp2_vec[:-1] * self.dx2)
            
        # Node N-1: Bottom (Adiabatic)
        k_last = k2_vec[-1]
        k_prev_last = k2_vec[-2]
        k_eff_last = self.k_harm(k_prev_last, k_last)
        q_in_last = k_eff_last * (T[self.N-2] - T[self.N-1]) / self.dx2
        dTdt[self.N-1] = q_in_last / (rho2_vec[-1] * Cp2_vec[-1] * (self.dx2/2))
        
        # --- History Equation (dT_max/dt) ---
        # Mathematica Logic: Tmax'[t] == Ramp[T'[t]] * SmoothStep[T[t] - Tmax[t]]
        # T_max grows only when T is rising (dTdt > 0) and T exceeds T_max.
        
        ramp_term = np.maximum(0.0, dTdt)
        switch_term = self.sigmoid(T - T_max, stiffness=100.0) # Smooth transition
        
        dTmax_dt = ramp_term * switch_term
        
        # Combine derivatives
        return np.concatenate([dTdt, dTmax_dt])

    def solve(self):
        t_span = (0, self.p['t_final'])
        
        # Initial Conditions
        T_init = np.ones(self.N) * self.p['T_init']
        Tmax_init = np.ones(self.N) * self.p['T_init']
        
        # State Vector = [T, Tmax] (Size: 2*N)
        y0 = np.concatenate([T_init, Tmax_init])
        
        # Use BDF (Stiff Solver)
        # Match Mathematica StartingStepSize -> 0.0001
        sol = solve_ivp(
            self.ode_system,
            t_span,
            y0,
            method='BDF',
            rtol=1e-6,
            atol=1e-8,
            first_step=1e-4,
            jac_sparsity=self.jac_sparsity,
        )
        
        # Result splitting
        # sol.y shape is (2N, time_steps)
        sol.T = sol.y[:self.N, :]
        sol.Tmax = sol.y[self.N:, :]
        
        return sol

# =============================================================================
# 2. Table Editor (GUI Utility - No Changes)
# =============================================================================
class TableEditor(ctk.CTkToplevel):
    def __init__(self, parent, title, data_list, headers, on_save_callback, default_key=None):
        super().__init__(parent)
        self.title(title)
        self.geometry("500x500")
        self.transient(parent)
        self.lift()
        self.focus_force()
        self.data_list = copy.deepcopy(data_list)
        self.headers = headers
        self.on_save = on_save_callback
        self.default_key = default_key
        self.entries = []

        self.btn_frame = ctk.CTkFrame(self)
        self.btn_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(
            self.btn_frame,
            text="Add Row",
            command=self.add_row,
            width=90,
            fg_color=COLOR_BTN_SOFT_BLUE,
            hover_color=COLOR_BTN_SOFT_BLUE_HOVER,
            text_color="white",
            font=UI_FONT_BUTTON,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.btn_frame,
            text="Save Default",
            command=self.save_default,
            width=110,
            fg_color=COLOR_BTN_SOFT_BLUE,
            hover_color=COLOR_BTN_SOFT_BLUE_HOVER,
            text_color="white",
            font=UI_FONT_BUTTON,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.btn_frame,
            text="Plot Graph",
            command=self.show_graph,
            width=90,
            fg_color="#3A3A3C",
            hover_color="#48484A",
            text_color="white",
            font=UI_FONT_BUTTON,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.btn_frame,
            text="Save & Close",
            command=self.save_data,
            fg_color=COLOR_BTN_SOFT_GREEN,
            hover_color=COLOR_BTN_SOFT_GREEN_HOVER,
            text_color="white",
            width=110,
            font=UI_FONT_BUTTON,
        ).pack(side="right", padx=5)

        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)

        for col, text in enumerate(headers):
            lbl = ctk.CTkLabel(self.scroll_frame, text=text, font=UI_FONT_SECTION)
            lbl.grid(row=0, column=col, padx=5, pady=5)
        self.refresh_grid()

    def refresh_grid(self):
        for entry in self.entries: entry.destroy()
        self.entries = []
        for r, row_data in enumerate(self.data_list):
            row_entries = []
            for c, val in enumerate(row_data):
                entry = ctk.CTkEntry(self.scroll_frame, width=80)
                entry.grid(row=r+1, column=c, padx=2, pady=2)
                entry.insert(0, str(val))
                row_entries.append(entry)
            del_btn = ctk.CTkButton(
                self.scroll_frame,
                text="X",
                width=30,
                fg_color=COLOR_BTN_SOFT_RED,
                hover_color=COLOR_BTN_SOFT_RED_HOVER,
                text_color="white",
                command=lambda idx=r: self.delete_row(idx),
            )
            del_btn.grid(row=r+1, column=len(self.headers), padx=5)
            self.entries.extend(row_entries + [del_btn])

    def add_row(self):
        self.data_list.append([0.0] * len(self.headers))
        self.refresh_grid()

    def delete_row(self, idx):
        del self.data_list[idx]
        self.refresh_grid()

    def _get_data_from_ui(self):
        new_data = []
        cols = len(self.headers)
        children = self.scroll_frame.winfo_children()
        widgets = children[cols:] 
        current_row = []
        for widget in widgets:
            if isinstance(widget, ctk.CTkEntry):
                try: val = float(widget.get())
                except: val = 0.0
                current_row.append(val)
                if len(current_row) == cols:
                    new_data.append(current_row)
                    current_row = []
        new_data.sort(key=lambda x: x[0])
        return new_data

    def show_graph(self):
        data = self._get_data_from_ui()
        if not data:
            _show_msg("No Data", "No data to plot.", "warning")
            return
        
        # Prepare data
        arr = np.array(data)
        x = arr[:, 0]
        y_cols = arr[:, 1:]
        x_label = self.headers[0]
        y_labels = self.headers[1:]
        
        # Popup Window
        top = ctk.CTkToplevel(self)
        top.title(f"Graph Preview: {self.title}")
        top.geometry("600x800")
        top.transient(self)
        top.lift()
        top.focus_force()
        
        # Plot
        num_plots = len(y_labels)
        fig, axes = plt.subplots(num_plots, 1, figsize=(5, 3*num_plots), sharex=True)
        if num_plots == 1: axes = [axes]
        
        fig.patch.set_facecolor('#242426')
        for i, ax in enumerate(axes):
            ax.plot(x, y_cols[:, i], 'o-', color='#0A84FF', linewidth=1.5, markersize=4)
            ax.set_ylabel(y_labels[i], color='#F2F2F7')
            ax.grid(True, color='#48484A', linestyle='--')
            ax.set_facecolor('#2C2C2E')
            ax.tick_params(colors='#F2F2F7')
            for spine in ax.spines.values(): spine.set_edgecolor('#F2F2F7')

        axes[-1].set_xlabel(x_label, color='#F2F2F7')
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def save_data(self):
        new_data = self._get_data_from_ui()
        self.on_save(new_data)
        self.destroy()

    def save_default(self):
        if not self.default_key:
            _show_msg("Warning", "No default key is set for this editor.", "warning")
            return

        new_data = self._get_data_from_ui()
        if not new_data:
            _show_msg("Warning", "No data to save.", "warning")
            return
        # Keep in-memory data consistent with the saved defaults
        self.on_save(new_data)

        try:
            self.master.save_default_dataset(self.default_key, self.headers, new_data)
            defaults_path = get_user_defaults_path()
            if self.default_key in LINKED_CORE_KEYS:
                _show_msg(
                    "Success",
                    "Saved linked core defaults (cork/cork_no_pyro/mass).\n"
                    f"{defaults_path}",
                )
            else:
                const_name = DEFAULT_KEY_MAP[self.default_key]
                _show_msg("Success", f"Saved as default: {const_name}\n{defaults_path}")
        except Exception as e:
            _show_msg("Error", f"Failed to write user defaults: {e}", "error")

# =============================================================================
# 3. Graph Settings Window
# =============================================================================
class GraphSettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent, lines_data, mode, on_save_callback):
        super().__init__(parent)
        self.mode = mode  # 'node', 'delta', 'time'
        self.title(f"Graph Options - {mode.capitalize()}")
        self.geometry("550x400")
        self.transient(parent)
        self.lift()
        
        self.lines_data = copy.deepcopy(lines_data)
        self.on_save = on_save_callback
        self.rows = []

        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=5)
        
        if self.mode == 'delta':
            ctk.CTkLabel(header_frame, text="Node A", width=70, anchor="w", font=UI_FONT_LABEL).pack(side="left", padx=5)
            ctk.CTkLabel(header_frame, text="Node B", width=70, anchor="w", font=UI_FONT_LABEL).pack(side="left", padx=5)
        else:
            lbl_text = "Time (s)" if self.mode == 'time' else "Node Number"
            ctk.CTkLabel(header_frame, text=lbl_text, width=120, anchor="w", font=UI_FONT_LABEL).pack(side="left", padx=5)
            
        ctk.CTkLabel(header_frame, text="Color", width=80, anchor="w", font=UI_FONT_LABEL).pack(side="left", padx=5)
        ctk.CTkLabel(header_frame, text="Label", width=100, anchor="w", font=UI_FONT_LABEL).pack(side="left", padx=5)

        # Scrollable Area
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.refresh_rows()

        # Hint
        if self.mode == 'time':
            hint = "* Enter time in seconds (e.g. 0, 60, 120)"
        elif self.mode == 'delta':
            hint = "* Delta = Node A - Node B (e.g. 20, 59)"
        else:
            hint = "* Node Index: 0, 20, 59 (Last)"
        ctk.CTkLabel(self, text=hint, font=UI_FONT_LABEL, text_color="gray").pack(pady=(5, 0))

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(btn_frame, text="Add Line", command=self.add_line, width=100, font=UI_FONT_BUTTON).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Apply", command=self.save, width=100, fg_color=COLOR_BTN_SOFT_GREEN, text_color="white", hover_color=COLOR_BTN_SOFT_GREEN_HOVER, font=UI_FONT_BUTTON).pack(side="right", padx=5)

    def _get_sort_key(self, line):
        try:
            if self.mode == 'delta':
                return str(line.get('node_a', ''))
            elif self.mode == 'time':
                val = str(line.get('time', '0')).strip().lower()
                if val == 'start': return -1.0
                if val == 'end': return float('inf')
                return float(val) if val.replace('.','',1).isdigit() else 0.0
            else:
                val = str(line.get('node', '0')).strip().lower()
                if val.startswith('surf'): return -1
                ival = int(val)
                return float('inf') if ival < 0 else ival
        except:
            return float('inf')

    def refresh_rows(self):
        self.lines_data.sort(key=self._get_sort_key)
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.rows = []
        
        for i, line in enumerate(self.lines_data):
            row_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)
            
            if self.mode == 'delta':
                ent_a = ctk.CTkEntry(row_frame, width=70)
                ent_a.pack(side="left", padx=5)
                ent_a.insert(0, str(line.get('node_a', '0')))
                
                ent_b = ctk.CTkEntry(row_frame, width=70)
                ent_b.pack(side="left", padx=5)
                ent_b.insert(0, str(line.get('node_b', '-1')))
                row_data = {'node_a': ent_a, 'node_b': ent_b}
            else:
                key = 'time' if self.mode == 'time' else 'node'
                ent_main = ctk.CTkEntry(row_frame, width=120)
                ent_main.pack(side="left", padx=5)
                ent_main.insert(0, str(line.get(key, '0')))
                row_data = {key: ent_main}
            
            ent_color = ctk.CTkEntry(row_frame, width=65)
            ent_color.pack(side="left", padx=(5, 2))
            color_val = str(line.get('color', '#ffffff'))
            ent_color.insert(0, color_val)
            
            btn_pick = ctk.CTkButton(row_frame, text="", width=24, height=24, fg_color=color_val, border_width=1, border_color="#3A3A3C")
            btn_pick.configure(command=lambda e=ent_color, b=btn_pick: self.pick_color(e, b))
            btn_pick.pack(side="left", padx=(0, 5))
            
            ent_label = ctk.CTkEntry(row_frame, width=100)
            ent_label.pack(side="left", padx=5)
            ent_label.insert(0, str(line.get('label', '')))
            
            btn_del = ctk.CTkButton(row_frame, text="X", width=30, fg_color=COLOR_BTN_SOFT_RED, text_color="white", hover_color=COLOR_BTN_SOFT_RED_HOVER, command=lambda idx=i: self.delete_line(idx))
            btn_del.pack(side="left", padx=5)
            
            row_data.update({'color': ent_color, 'label': ent_label})
            self.rows.append(row_data)

    def pick_color(self, entry, btn):
        curr = entry.get()
        try:
            color = askcolor(color=curr, title="Select Color", parent=self)[1]
        except:
            color = askcolor(title="Select Color", parent=self)[1]
            
        if color:
            entry.delete(0, "end")
            entry.insert(0, color)
            btn.configure(fg_color=color)

    def add_line(self):
        self._update_data_from_ui()
        if self.mode == 'delta':
            self.lines_data.append({'node_a': 'mid', 'node_b': '-1', 'color': '#ffffff', 'label': 'Delta'})
        elif self.mode == 'time':
            self.lines_data.append({'time': 'mid', 'color': '#ffffff', 'label': 't=Mid'})
        else:
            self.lines_data.append({'node': '0', 'color': '#ffffff', 'label': 'Node'})
        self.refresh_rows()

    def delete_line(self, idx):
        self._update_data_from_ui()
        del self.lines_data[idx]
        self.refresh_rows()

    def _update_data_from_ui(self):
        new_data = []
        for row in self.rows:
            item = {'color': row['color'].get(), 'label': row['label'].get()}
            if self.mode == 'delta':
                item.update({'node_a': row['node_a'].get(), 'node_b': row['node_b'].get()})
            else:
                key = 'time' if self.mode == 'time' else 'node'
                item[key] = row.get(key).get()
            new_data.append(item)
        self.lines_data = new_data

    def save(self):
        self._update_data_from_ui()
        self.on_save(self.lines_data)
        self.destroy()

# =============================================================================
# 4. GUI Main
# =============================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

UNIT_PRESETS = {
    "SI (m, kg, W)": {
        "len": "m", "h": "W/m^2K", "k": "W/mK", "rho": "kg/m^3", "cp": "J/kgK",
        "temp": "degC", "time": "s", "def_L1": "0.002", "def_L2": "0.004", "def_cp_char": "500", "def_N2": "5"
    }
}

class HeatTransferApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HTP50cork")
        self.geometry("1300x900")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Menubar
        menu_bar = ctk.CTkFrame(self, height=32, corner_radius=0)
        menu_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        menu_bar.grid_columnconfigure(1, weight=1)

        
        self.mat_data = {'cork': [], 'cork_no_pyro': [], 'mass': [], 'metal': []} 
        self.bc_data = {'h': [], 'Tr': []}
        
        self.last_sol = None
        self.last_sol_no_pyro = None
        self.last_solver = None
        
        self.graph_configs = {
            'top': [
                {'node': '0', 'color': '#ff9999', 'label': 'Cork Surface'},
                {'node': '5', 'color': '#ffeaa7', 'label': 'Interface'},
                {'node': '9', 'color': '#81ecec', 'label': 'Bottom'}
            ],
            'mid': [
                {'node': '0', 'color': '#ff9999', 'label': 'Cp Cork Surface'},
                {'node': '5', 'color': '#ffeaa7', 'label': 'Cp Interface'}
            ],
            'bot': [
                {'node': '5', 'color': '#ffeaa7', 'label': 'Interface'},
                {'node': '9', 'color': '#81ecec', 'label': 'Bottom'}
            ]
        }

        # Track geometry for auto-updating graph nodes
        self.last_N1 = 5
        self.last_N2 = 5

        self.load_defaults()
        self.entries = {}
        self.labels = {} 

        # Sidebar setup
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(15, weight=1)

        self.create_unit_selector(0)
        self.create_input_group("Geometry", 1, [
            ("L1", "def_L1"), ("L2", "def_L2"), ("N1 (Nodes Cork)", "5"), ("N2 (Nodes Metal)", "def_N2"),
            ("t_final (s)", "120"), ("T_init (degC)", "63")
        ])
        self.create_pyrolysis_group(2)
        self.create_data_manager_group("Material & BC Data", 3)

        self.run_button = ctk.CTkButton(
            self.sidebar,
            text="RUN SIMULATION",
            command=self.run_simulation,
            height=36,
            fg_color="#0A84FF",
            hover_color="#3399FF",
            text_color="white",
            font=UI_FONT_BUTTON,
        )
        self.run_button.grid(row=5, column=0, padx=20, pady=(14, 10))
        
        self.progress_bar = ctk.CTkProgressBar(self.sidebar, height=12, corner_radius=6)
        self.progress_bar.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.progress_bar.set(0.0)
        
        self.status_label = ctk.CTkLabel(self.sidebar, text="Ready", text_color="gray", font=UI_FONT_LABEL)
        self.status_label.grid(row=7, column=0, padx=20, pady=(4, 8))

        # Plot setup
        self.graph_frame = ctk.CTkFrame(self)
        self.graph_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        self.setup_charts()

        self.current_unit_key = "SI (m, kg, W)"
        self.change_unit("SI (m, kg, W)")
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.quit()
        self.destroy()

    def load_defaults(self):
        self.mat_data['cork'] = self.parse_csv_string(DEFAULT_CORK_CSV)
        self.mat_data['cork_no_pyro'] = self.parse_csv_string(DEFAULT_CORK_NO_PYRO_CSV)
        self.mat_data['mass'] = self.parse_csv_string(DEFAULT_P50_MASS_CSV)
        self.mat_data['metal'] = self.parse_csv_string(DEFAULT_METAL_CSV)
        self.bc_data['h'] = self.parse_csv_string(DEFAULT_H_CSV)
        self.bc_data['Tr'] = self.parse_csv_string(DEFAULT_TR_CSV)
        self._load_user_defaults()

    def _load_user_defaults(self):
        defaults_path = get_user_defaults_path()
        if not os.path.exists(defaults_path):
            return

        try:
            with open(defaults_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return

        if not isinstance(payload, dict):
            return

        is_current_payload = payload.get(USER_DEFAULTS_VERSION_KEY) == USER_DEFAULTS_VERSION

        # Apply linked core data only for current-version payloads and only as a complete set.
        if is_current_payload:
            linked_parsed = {}
            linked_ok = True
            for key in LINKED_CORE_KEYS:
                const_name = DEFAULT_KEY_MAP[key]
                csv_str = payload.get(const_name)
                if not isinstance(csv_str, str):
                    linked_ok = False
                    break
                try:
                    linked_parsed[key] = self.parse_csv_string(csv_str)
                except Exception:
                    linked_ok = False
                    break
            if linked_ok:
                for key, parsed in linked_parsed.items():
                    self.mat_data[key] = parsed

        # Independent datasets can be loaded individually (legacy payload compatible).
        for key in ("metal", "h", "Tr"):
            const_name = DEFAULT_KEY_MAP[key]
            csv_str = payload.get(const_name)
            if not isinstance(csv_str, str):
                continue
            try:
                parsed = self.parse_csv_string(csv_str)
            except Exception:
                continue
            if key == "metal":
                self.mat_data[key] = parsed
            else:
                self.bc_data[key] = parsed

    def setup_charts(self):
        self.charts = {}
        # Clear graph frame
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
        sections = [
            ('top', "Temperature History"),
            ('mid', "Specific Heat (Cp)"),
            ('bot', "Pyrolysis Comparison")
        ]
        
        self.graph_frame.grid_rowconfigure(0, weight=1)
        self.graph_frame.grid_rowconfigure(1, weight=1)
        self.graph_frame.grid_rowconfigure(2, weight=1)
        self.graph_frame.grid_columnconfigure(0, weight=1)

        for i, (key, title) in enumerate(sections):
            # Container
            frame = ctk.CTkFrame(self.graph_frame)
            frame.grid(row=i, column=0, sticky="nsew", padx=2, pady=2)
            frame.grid_rowconfigure(1, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            
            # Header
            header = ctk.CTkFrame(frame, height=28, fg_color="transparent")
            header.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
            
            ctk.CTkLabel(header, text=title, font=UI_FONT_SECTION).pack(side="left", padx=5)
            
            # Buttons
            ctk.CTkButton(header, text="Options", width=60, height=20, command=lambda k=key: self.open_graph_options(k)).pack(side="right", padx=2)
            ctk.CTkButton(header, text="PNG", width=40, height=20, command=lambda k=key: self.export_section_png(k)).pack(side="right", padx=2)
            ctk.CTkButton(header, text="CSV", width=40, height=20, command=lambda k=key: self.export_section_csv(k)).pack(side="right", padx=2)
            if key == 'top':
                ctk.CTkButton(
                    header,
                    text="Peak Temp",
                    width=74,
                    height=20,
                    command=self.show_top_peak_temperatures,
                ).pack(side="right", padx=2)
            
            # Canvas
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor('#242426')
            ax.set_facecolor('#2C2C2E')
            ax.tick_params(colors='#F2F2F7')
            for spine in ax.spines.values(): spine.set_edgecolor('#F2F2F7')
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
            
            self.charts[key] = {'fig': fig, 'ax': ax, 'canvas': canvas}

    def parse_csv_string(self, csv_str):
        data = []
        f = io.StringIO(csv_str)
        reader = csv.reader(f)
        next(reader, None) 
        for row in reader:
            if row: data.append([float(x) for x in row])
        return data

    def _csv_block_from_rows(self, headers, rows):
        header = ",".join(headers)
        body = "\n".join(",".join(f"{v:g}" for v in row) for row in rows)
        return header if not body else f"{header}\n{body}"

    def save_default_dataset(self, key, headers, new_data):
        const_name = DEFAULT_KEY_MAP.get(key)
        if not const_name:
            raise ValueError(f"Unsupported default key: {key}")

        defaults_path = get_user_defaults_path()
        payload = {}
        if os.path.exists(defaults_path):
            try:
                with open(defaults_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    payload = loaded
            except Exception:
                payload = {}

        payload[USER_DEFAULTS_VERSION_KEY] = USER_DEFAULTS_VERSION

        if key in LINKED_CORE_KEYS:
            # Persist the linked core as a coherent set to prevent partial override mismatches.
            core_data = {
                "cork": self.mat_data.get("cork", []),
                "cork_no_pyro": self.mat_data.get("cork_no_pyro", []),
                "mass": self.mat_data.get("mass", []),
            }
            core_data[key] = new_data
            for core_key, rows in core_data.items():
                payload[DEFAULT_KEY_MAP[core_key]] = self._csv_block_from_rows(
                    DEFAULT_HEADERS_BY_KEY[core_key], rows
                )
        else:
            payload[const_name] = self._csv_block_from_rows(headers, new_data)

        with open(defaults_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def create_unit_selector(self, row_idx):
        container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        container.grid(row=row_idx, column=0, padx=14, pady=(10, 4), sticky="ew")
        ctk.CTkLabel(
            container,
            text="Unit System",
            font=UI_FONT_SECTION,
            text_color=COLOR_SECTION_TEXT,
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

        frame = ctk.CTkFrame(container)
        frame.pack(fill="x", padx=2, pady=(0, 6))
        self.unit_var = ctk.StringVar(value="SI (m, kg, W)")
        combo = ctk.CTkComboBox(frame, values=list(UNIT_PRESETS.keys()), command=self.change_unit, variable=self.unit_var, font=UI_FONT_LABEL)
        combo.pack(fill="x", padx=5, pady=5)

    def change_unit(self, choice):
        u = UNIT_PRESETS[choice]
        key_map = {
            "L1": f"L1 (Cork [{u['len']}])", "L2": f"L2 (Metal [{u['len']}])",
            "Cp_char": f"Cp_char [{u['cp']}]"
        }
        
        if "L1" in self.entries: self.entries["L1"].delete(0, "end"); self.entries["L1"].insert(0, u['def_L1'])
        if "L2" in self.entries: self.entries["L2"].delete(0, "end"); self.entries["L2"].insert(0, u['def_L2'])
        if "N2" in self.entries: self.entries["N2"].delete(0, "end"); self.entries["N2"].insert(0, u['def_N2'])
        for key, text in key_map.items():
            if key in self.labels: self.labels[key].configure(text=text)
            
        self.current_unit_key = choice
        self._update_cp_char_from_cork_data()

    def _update_cp_char_from_cork_data(self):
        if "Cp_char" not in self.entries:
            return
        cork_data = self.mat_data.get('cork', [])
        if not cork_data:
            return
        # Use highest temperature row as "last"
        try:
            last_row = max(cork_data, key=lambda r: r[0])
            cp_val = float(last_row[2])
        except Exception:
            return
        self.entries["Cp_char"].delete(0, "end")
        self.entries["Cp_char"].insert(0, f"{cp_val:g}")

    def create_input_group(self, title, row_idx, fields):
        container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        container.grid(row=row_idx, column=0, padx=14, pady=(10, 4), sticky="ew")
        ctk.CTkLabel(
            container,
            text=title,
            font=UI_FONT_SECTION,
            text_color=COLOR_SECTION_TEXT,
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

        frame = ctk.CTkFrame(container)
        frame.pack(fill="x", padx=2, pady=(0, 6))
        for txt, val in fields:
            f = ctk.CTkFrame(frame, fg_color="transparent")
            f.pack(fill="x", padx=8, pady=4)
            key = txt.split(" ")[0] 
            lbl = ctk.CTkLabel(f, text=txt, width=140, anchor="w", font=UI_FONT_LABEL, text_color=COLOR_LABEL_TEXT)
            lbl.pack(side="left")
            self.labels[key] = lbl 
            entry = ctk.CTkEntry(f, height=26)
            entry.pack(side="right", expand=True, fill="x")
            if "def_" not in val: entry.insert(0, val)
            self.entries[key] = entry

    def create_pyrolysis_group(self, row_idx):
        container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        container.grid(row=row_idx, column=0, padx=14, pady=(10, 4), sticky="ew")
        ctk.CTkLabel(
            container,
            text="Pyrolysis (Cork P50)",
            font=UI_FONT_SECTION,
            text_color=COLOR_SECTION_TEXT,
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

        frame = ctk.CTkFrame(container)
        frame.pack(fill="x", padx=2, pady=(0, 6))

        mode_row = ctk.CTkFrame(frame, fg_color="transparent")
        mode_row.pack(fill="x", padx=8, pady=(6, 4))
        ctk.CTkLabel(
            mode_row,
            text="Density Change",
            width=140,
            anchor="w",
            font=UI_FONT_LABEL,
            text_color=COLOR_LABEL_TEXT,
        ).pack(side="left")
        self.pyro_mode_var = ctk.StringVar(value="advanced")
        self.pyro_mode_switch = ctk.CTkSwitch(
            mode_row,
            text="ON",
            variable=self.pyro_mode_var,
            onvalue="advanced",
            offvalue="simple",
            command=self._refresh_pyrolysis_inputs,
            font=UI_FONT_LABEL,
        )
        self.pyro_mode_switch.pack(side="right")

        # Mass profile row (shown only in advanced mode)
        self.mass_row = ctk.CTkFrame(frame, fg_color="transparent")
        ctk.CTkLabel(
            self.mass_row, text="Mass Profile", width=140, anchor="w",
            font=UI_FONT_LABEL, text_color=COLOR_LABEL_TEXT,
        ).pack(side="left")
        ctk.CTkButton(
            self.mass_row, text="Load", width=64, height=26,
            command=lambda: self.load_csv('mass'),
            fg_color="#3A3A3C", hover_color="#48484A", font=UI_FONT_LABEL,
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            self.mass_row, text="Edit", width=64, height=26,
            command=lambda: self.open_editor('mass', ["Temp (degC)", "MassNorm (%)"]),
            fg_color="#3A3A3C", hover_color="#48484A", font=UI_FONT_LABEL,
        ).pack(side="left", padx=2)

        self.pyro_rows = {}
        self._create_pyro_input_row(frame, "T_onset", "T_onset (degC)", "156.85")
        self._create_pyro_input_row(frame, "Cp_char", "Cp_char", "")
        self._create_pyro_input_row(frame, "T_simple", "T_critical (degC)", "500")
        self._refresh_pyrolysis_inputs()

    def _create_pyro_input_row(self, parent, key, label_text, default_value):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        lbl = ctk.CTkLabel(row, text=label_text, width=140, anchor="w", font=UI_FONT_LABEL, text_color=COLOR_LABEL_TEXT)
        lbl.pack(side="left")
        entry = ctk.CTkEntry(row, height=26)
        entry.pack(side="right", expand=True, fill="x")
        if default_value:
            entry.insert(0, default_value)
        self.pyro_rows[key] = row
        self.labels[key] = lbl
        self.entries[key] = entry

    def _refresh_pyrolysis_inputs(self):
        mode = self.pyro_mode_var.get() if hasattr(self, "pyro_mode_var") else "advanced"

        for key in ("T_onset", "Cp_char", "T_simple"):
            row = self.pyro_rows.get(key)
            if row is not None:
                row.pack_forget()

        if hasattr(self, "mass_row"):
            self.mass_row.pack_forget()

        # Show mass profile only in advanced mode (toggle ON)
        if mode == "advanced" and hasattr(self, "mass_row"):
            self.mass_row.pack(fill="x", padx=8, pady=4)

        # Both modes use T_critical (T_simple)
        self.pyro_rows["T_simple"].pack(fill="x", padx=8, pady=4)

    def create_data_manager_group(self, title, row_idx):
        container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        container.grid(row=row_idx, column=0, padx=14, pady=(10, 4), sticky="ew")
        ctk.CTkLabel(
            container,
            text=title,
            font=UI_FONT_SECTION,
            text_color=COLOR_SECTION_TEXT,
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

        frame = ctk.CTkFrame(container)
        frame.pack(fill="x", padx=2, pady=(0, 6))
        def add_btn(txt, cmd_load, cmd_edit):
            f = ctk.CTkFrame(frame, fg_color="transparent")
            f.pack(fill="x", padx=8, pady=4)
            ctk.CTkLabel(f, text=txt, width=140, anchor="w", font=UI_FONT_LABEL, text_color=COLOR_LABEL_TEXT).pack(side="left")
            ctk.CTkButton(f, text="Load", width=64, height=26, command=cmd_load, fg_color="#3A3A3C", hover_color="#48484A", font=UI_FONT_LABEL).pack(side="left", padx=2)
            ctk.CTkButton(f, text="Edit", width=64, height=26, command=cmd_edit, fg_color="#3A3A3C", hover_color="#48484A", font=UI_FONT_LABEL).pack(side="left", padx=2)
        add_btn("Cork (Pyrolysis)", lambda: self.load_csv('cork'), lambda: self.open_editor('cork', ["Temp (degC)", "k", "Cp", "rho"]))
        add_btn("Cork (No Pyrolysis)", lambda: self.load_csv('cork_no_pyro'), lambda: self.open_editor('cork_no_pyro', ["Temp (degC)", "k", "Cp", "rho"]))
        add_btn("Metal", lambda: self.load_csv('metal'), lambda: self.open_editor('metal', ["Temp (degC)", "k", "Cp", "rho"]))
        add_btn("h(t)", lambda: self.load_csv('h'), lambda: self.open_editor('h', ["Time", "h"]))
        add_btn("Tr(t)", lambda: self.load_csv('Tr'), lambda: self.open_editor('Tr', ["Time", "Tr (degC)"]))

    def load_csv(self, key):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filename: return
        try:
            data = []
            with open(filename, newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row: data.append([float(x) for x in row])
            if key in ['cork', 'cork_no_pyro', 'mass', 'metal']: self.mat_data[key] = data
            else: self.bc_data[key] = data
            _show_msg("Success", f"{key} loaded!")
            if key == 'cork':
                self._update_cp_char_from_cork_data()
        except Exception as e: _show_msg("Error", str(e), "error")

    def open_graph_options(self, target):
        mode_map = {'top': 'node', 'mid': 'node', 'bot': 'node'}
        if target not in self.graph_configs: return
        
        # Calculate current max node index to replace -1
        try:
            n1 = int(self.entries['N1'].get())
            n2 = int(self.entries['N2'].get())
            current_max = str(n1 + n2 - 1)
        except:
            current_max = '59'

        # Create a copy and replace -1 or 'bot' with actual number
        config_to_pass = copy.deepcopy(self.graph_configs[target])
        for line in config_to_pass:
            for key in ['node', 'node_a', 'node_b']:
                if key in line:
                    val = str(line[key]).strip().lower()
                    if val == '-1' or val == 'bot':
                        line[key] = current_max

        GraphSettingsWindow(self, config_to_pass, mode_map[target], 
                            lambda new_data: self.update_graph_config(target, new_data))

    def update_graph_config(self, target, new_lines):
        self.graph_configs[target] = new_lines
        if self.last_sol and self.last_solver:
            self.update_plots(self.last_solver, self.last_sol, self.last_sol_no_pyro)

    def open_editor(self, key, headers):
        if key in ['cork', 'cork_no_pyro', 'mass', 'metal']: current_data = self.mat_data[key]
        else: current_data = self.bc_data[key]
        def save_callback(new_data):
            if key in ['cork', 'cork_no_pyro', 'mass', 'metal']: self.mat_data[key] = new_data
            else: self.bc_data[key] = new_data
            if key == 'cork':
                self._update_cp_char_from_cork_data()
        TableEditor(self, f"Edit: {key}", current_data, headers, save_callback, default_key=key)

    def run_simulation(self):
        try:
            # Cp_char is always derived from cork (pyro) last Cp
            self._update_cp_char_from_cork_data()
            # 1. Read parameters (main thread)
            n1_val = int(self.entries['N1'].get())
            n2_val = int(self.entries['N2'].get())

            # Auto-update graph nodes if geometry changed
            if n1_val != self.last_N1 or n2_val != self.last_N2:
                old_int = self.last_N1
                old_bot = self.last_N1 + self.last_N2 - 1
                new_int = n1_val
                new_bot = n1_val + n2_val - 1
                
                for key in self.graph_configs:
                    for line in self.graph_configs[key]:
                        if 'node' in line and str(line['node']).isdigit():
                            v = int(line['node'])
                            if v == old_int: line['node'] = str(new_int)
                            elif v == old_bot: line['node'] = str(new_bot)
                self.last_N1 = n1_val
                self.last_N2 = n2_val

            params = {
                'L1': float(self.entries['L1'].get()),
                'L2': float(self.entries['L2'].get()),
                'N1': n1_val,
                'N2': n2_val,
                't_final': float(self.entries['t_final'].get()),
                'T_init': float(self.entries['T_init'].get()),
                'T_onset': float(self.entries['T_onset'].get()), 
                'Cp_char': float(self.entries['Cp_char'].get()),
                'T_simple': float(self.entries['T_simple'].get()),
                'pyro_mode': self.pyro_mode_var.get() if hasattr(self, 'pyro_mode_var') else 'advanced',
            }
            
            # 1. Solver for Pyrolysis (Standard)
            mat_data_pyro = {'cork': self.mat_data['cork'], 'mass': self.mat_data['mass'], 'metal': self.mat_data['metal']}
            solver_pyro = HeatSolver(params, mat_data_pyro, self.bc_data, enable_pyrolysis=True)
            
            # 2. Solver for No Pyrolysis (Comparison)
            mat_data_no_pyro = {'cork': self.mat_data['cork_no_pyro'], 'mass': self.mat_data['mass'], 'metal': self.mat_data['metal']}
            solver_no_pyro = HeatSolver(params, mat_data_no_pyro, self.bc_data, enable_pyrolysis=False)
            
        except Exception as e:
            _show_msg("Input Error", f"Invalid Input: {e}", "error")
            return

        # 2. Update UI state
        self.status_label.configure(text="Solving...", text_color=COLOR_WARN)
        self.run_button.configure(state="disabled")
        self.progress_bar.set(0.0)
        
        # 3. Run calculations in a worker thread
        thread = threading.Thread(target=self._run_solver_thread, args=(solver_pyro, solver_no_pyro))
        thread.daemon = True
        thread.start()
        
        self.monitor_progress(solver_pyro, thread)

    def monitor_progress(self, solver, thread):
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return

        if thread.is_alive():
            t_curr = getattr(solver, 't_current', 0.0)
            t_final = solver.p['t_final']
            if t_final > 1e-9:
                self.progress_bar.set(min(t_curr / t_final, 1.0))
            self.after(100, lambda: self.monitor_progress(solver, thread))
        else:
            self.progress_bar.set(1.0)

    def _run_solver_thread(self, solver1, solver2):
        try:
            start_t = time.time()
            sol1 = solver1.solve()
            sol2 = solver2.solve()
            elapsed = time.time() - start_t
            # Dispatch completion callback on the main thread
            self.after(0, lambda: self._on_simulation_complete(solver1, sol1, sol2, elapsed))
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self._on_simulation_error(err_msg))

    def _on_simulation_complete(self, solver, sol, sol_no_pyro, elapsed):
        try:
            if not self.winfo_exists(): return
        except Exception: return

        self.last_solver = solver
        self.last_sol = sol
        self.last_sol_no_pyro = sol_no_pyro
        self.update_plots(solver, sol, sol_no_pyro)
        self.status_label.configure(text=f"Done ({elapsed:.2f}s)", text_color="#00ff00")
        self.run_button.configure(state="normal")

    def _on_simulation_error(self, error_msg):
        try:
            if not self.winfo_exists(): return
        except Exception: return

        self.status_label.configure(text="Error", text_color="red")
        _show_msg("Simulation Error", error_msg, "error")
        print(error_msg)
        self.run_button.configure(state="normal")

    def export_section_csv(self, key):
        if self.last_sol is None:
            _show_msg("Warning", "No simulation results to export.\nPlease run the simulation first.", "warning")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile=f"{key}_data.csv")
        if not filename:
            return

        try:
            t = self.last_sol.t
            T = self.last_sol.T
            Tmax = getattr(self.last_sol, 'Tmax', np.zeros_like(T))
            solver = self.last_solver
            
            # Collect data columns based on graph config
            columns = [("Time", t)]
            
            for line in self.graph_configs[key]:
                try:
                    node_val = line['node']
                    idx = self._parse_node_idx(node_val, solver, T.shape[0])
                    if idx >= T.shape[0] or idx < -T.shape[0]: continue
                    if idx < 0: idx = T.shape[0] + idx
                    
                    label = line['label']
                    
                    if key == 'mid': # Cp data
                        T_node = T[idx, :]
                        if idx <= solver.m:
                            Tmax_node = Tmax[idx, :]
                            _, val_data, _ = solver.get_props1(T_node, Tmax_node)
                        else:
                            _, val_data, _ = solver.get_props2(T_node)
                    else: # Temperature data
                        val_data = T[idx, :]
                        
                    columns.append((label, val_data))
                except:
                    pass
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [col[0] for col in columns]
                writer.writerow(header)
                for i in range(len(t)):
                    row = [col[1][i] for col in columns]
                    writer.writerow(row)
            
            _show_msg("Success", f"Results exported to:\n{filename}")
        except Exception as e:
            _show_msg("Export Error", str(e), "error")

    def export_section_png(self, key):
        if self.last_sol is None:
            _show_msg("Warning", "No simulation results to export.", "warning")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")], initialfile=f"{key}_graph.png")
        if not filename:
            return
            
        try:
            # Create a temporary figure for export (Light Theme)
            temp_fig, temp_ax = plt.subplots(figsize=(8, 5))
            
            # Plot data using the helper
            self._plot_graph_on_ax(temp_ax, key, self.last_solver, self.last_sol, self.last_sol_no_pyro, theme='light')
            
            # Save and close
            temp_fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(temp_fig)
            
            _show_msg("Success", f"Image exported to:\n{filename}")
        except Exception as e:
            _show_msg("Export Error", str(e), "error")

    def update_plots(self, solver, sol, sol_no_pyro=None):
        for key in ['top', 'mid', 'bot']:
            ax = self.charts[key]['ax']
            canvas = self.charts[key]['canvas']
            
            self._plot_graph_on_ax(ax, key, solver, sol, sol_no_pyro, theme='dark')
            canvas.draw()

    def show_top_peak_temperatures(self):
        if self.last_sol is None or self.last_solver is None:
            _show_msg("Warning", "No simulation results to inspect.\nPlease run the simulation first.", "warning")
            return

        solver = self.last_solver
        sol = self.last_sol
        t = sol.t
        T = sol.T
        u = UNIT_PRESETS[self.unit_var.get()]
        temp_unit = u.get('temp', 'degC')
        time_unit = u.get('time', 's')

        rows = []
        for line in self.graph_configs.get('top', []):
            try:
                node_val = line.get('node', 0)
                idx = self._parse_node_idx(node_val, solver, T.shape[0])
                if idx >= T.shape[0] or idx < -T.shape[0]:
                    continue
                if idx < 0:
                    idx = T.shape[0] + idx

                y = T[idx, :]
                peak_i = int(np.argmax(y))
                peak_t = float(t[peak_i])
                peak_temp = float(y[peak_i])
                label = line.get('label') or f"Node {idx}"
                rows.append(f"{label}: {peak_temp:.2f} {temp_unit} (t={peak_t:.2f} {time_unit})")
            except Exception:
                continue

        if not rows:
            _show_msg("Warning", "No valid line is configured in the first graph.", "warning")
            return

        _show_msg("Peak Temperatures", "\n".join(rows))

    def _plot_graph_on_ax(self, ax, key, solver, sol, sol_no_pyro, theme='dark'):
        ax.clear()
        
        # Theme settings
        if theme == 'dark':
            bg_color = '#2C2C2E'
            fg_color = '#F2F2F7'
            grid_color = '#48484A'
        else:
            bg_color = 'white'
            fg_color = 'black'
            grid_color = '#cccccc'
            
        ax.set_facecolor(bg_color)
        ax.grid(True, color=grid_color, linestyle='--')
        ax.tick_params(colors=fg_color)
        for spine in ax.spines.values(): spine.set_edgecolor(fg_color)
        ax.xaxis.label.set_color(fg_color)
        ax.yaxis.label.set_color(fg_color)
        ax.title.set_color(fg_color)

        # Disable scientific offset notation like '1e-6+3e2' on the y-axis.
        # Keep labels in plain numeric format for readability.
        ax.ticklabel_format(axis='y', style='plain')

        u = UNIT_PRESETS[self.unit_var.get()]
        temp_unit = u.get('temp', 'degC')
        time_unit = u.get('time', 's')

        t = sol.t
        T = sol.T
        
        if key == 'top':
            for line in self.graph_configs['top']:
                try:
                    node_val = line['node']
                    idx = self._parse_node_idx(node_val, solver, T.shape[0])
                    if idx >= T.shape[0] or idx < -T.shape[0]: continue
                    linestyle = '--' if str(node_val) == '-1' or str(node_val).lower() == 'bot' else '-'
                    ax.plot(t, T[idx, :], color=line['color'], label=line['label'], linewidth=1.5, linestyle=linestyle)
                except: pass
            ax.set_title("Temperature")
            ax.set_ylabel(f"Temperature [{temp_unit}]")
            ax.set_xlabel(f"Time [{time_unit}]")
            
        elif key == 'mid':
            Tmax = getattr(sol, 'Tmax', np.zeros_like(T))
            for line in self.graph_configs['mid']:
                try:
                    node_val = line['node']
                    idx = self._parse_node_idx(node_val, solver, T.shape[0])
                    if idx >= T.shape[0] or idx < -T.shape[0]: continue
                    if idx < 0: idx = T.shape[0] + idx
                    T_node = T[idx, :]
                    if idx <= solver.m:
                        Tmax_node = Tmax[idx, :]
                        _, Cp_node, _ = solver.get_props1(T_node, Tmax_node)
                    else:
                        _, Cp_node, _ = solver.get_props2(T_node)
                    ax.plot(t, Cp_node, color=line['color'], label=line['label'], linewidth=1.5)
                except: pass
            ax.set_title("Specific Heat (Cp)")
            ax.set_ylabel(f"Cp [{u.get('cp', 'J/kgK')}]")
            ax.set_xlabel(f"Time [{time_unit}]")
            
        elif key == 'bot':
            if sol_no_pyro:
                t_np = sol_no_pyro.t
                T_np = sol_no_pyro.T
            else:
                t_np, T_np = None, None
            for line in self.graph_configs['bot']:
                try:
                    node_val = line['node']
                    idx = self._parse_node_idx(node_val, solver, T.shape[0])
                    if idx >= T.shape[0] or idx < -T.shape[0]: continue
                    ax.plot(t, T[idx, :], color=line['color'], label=line['label'], linewidth=1.5)
                    if T_np is not None:
                        ax.plot(t_np, T_np[idx, :], color=line['color'], linestyle='--', alpha=0.7, linewidth=1.2)
                except: pass
            ax.set_title("Pyrolysis Effect (Solid: Pyro, Dashed: No-Pyro)")
            ax.set_xlabel(f"Time [{time_unit}]")
            ax.set_ylabel(f"Temperature [{temp_unit}]")

        legend = ax.legend(facecolor=bg_color, labelcolor=fg_color)
        if theme == 'light':
            legend.get_frame().set_edgecolor('#cccccc')
        
        ax.figure.tight_layout()

    def _parse_node_idx(self, val, solver, max_n):
        s = str(val).lower().strip()
        if s in ['mid', 'int', 'metal']: return solver.m
        if s in ['surf', 'cork']: return 0
        if s == 'bot': return -1
        try: return int(val)
        except: return 0

if __name__ == "__main__":
    app = HeatTransferApp()
    app.mainloop()
