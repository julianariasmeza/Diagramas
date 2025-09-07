# beam_model.py
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from string import Template

# =========================
# Entidades básicas
# =========================
@dataclass
class Support:
    """Apoyo en x con tipo: 'pinned', 'roller' o 'fixed'."""
    position: float
    kind: str  # 'pinned' | 'roller' | 'fixed'

@dataclass
class PointLoad:
    """Carga puntual: magnitude>0 hacia arriba (convención)."""
    position: float  # coordenada x
    magnitude: float  # fuerza vertical

@dataclass
class DistributedLinearLoad:
    """
    Carga distribuida lineal: w>0 hacia arriba (convención).
    Definida en [x_start, x_end] con intensidades w_start y w_end.
    """
    x_start: float
    x_end: float
    w_start: float
    w_end: float

    def __post_init__(self):
        if self.x_end <= self.x_start:
            raise ValueError("x_end debe ser mayor que x_start")

    def w(self, x):
        """Intensidad w(x) del tramo; fuera de [x_start, x_end] → 0."""
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        L = self.x_end - self.x_start
        if L <= 0:
            return out
        a = (self.w_end - self.w_start) / L
        b = self.w_start - a * self.x_start
        mask = (x >= self.x_start) & (x <= self.x_end)
        out[mask] = a * x[mask] + b
        return out

    def resultant_and_centroid(self) -> Tuple[float, float]:
        """
        Resultante R y abscisa del centroide xc del trapecio lineal sobre [x_start, x_end].
        R = 0,5(wa+wb)L; xc = x_start + L(wa+2wb)/(3(wa+wb)) si wa+wb!=0; si no, al centro.
        """
        L = self.x_end - self.x_start
        wa, wb = self.w_start, self.w_end
        R = 0.5 * (wa + wb) * L
        if abs(wa + wb) < 1e-12:
            xc = 0.5 * (self.x_start + self.x_end)
        else:
            xc = self.x_start + L * (wa + 2.0 * wb) / (3.0 * (wa + wb))
        return R, xc

@dataclass
class AppliedMoment:
    """Momento aplicado en x: CCW positivo (convención)."""
    position: float
    magnitude: float  # par

# =========================
# Analizador
# =========================
@dataclass
class BeamAnalyzer:
    """
    Analizador de viga estáticamente determinada:
    - Apoyos: 'pinned'/'roller' (reacción vertical) y/o 'fixed' (reacción vertical + momento).
    - Cargas: puntuales, distribuidas lineales por tramos y momentos aplicados.
    Convenciones: fuerzas y w>0 hacia arriba; momentos CCW positivos.
    """
    length: float
    units: Dict[str, str] = field(default_factory=lambda: {"x": "m", "w": "kN/m", "V": "kN", "M": "kN·m"})
    npts: int = 1201
    supports: List[Support] = field(default_factory=list)
    point_loads: List[PointLoad] = field(default_factory=list)
    dist_loads: List[DistributedLinearLoad] = field(default_factory=list)
    moments: List[AppliedMoment] = field(default_factory=list)

    # Resultados de reacciones
    reactions: Dict[float, float] = field(default_factory=dict)  # x -> R
    fix_moment: Dict[float, float] = field(default_factory=dict)  # x -> M

    # Campos discretos
    x: np.ndarray = field(init=False, repr=False)
    w_field: np.ndarray = field(init=False, repr=False)
    V: np.ndarray = field(init=False, repr=False)
    M: np.ndarray = field(init=False, repr=False)

    # ===== API de construcción =====
    def set_supports(self, supports: List[Support]):
        self.supports = sorted(supports, key=lambda s: s.position)

    def add_point_load(self, L: PointLoad): 
        self.point_loads.append(L)

    def add_distributed_load(self, W: DistributedLinearLoad): 
        self.dist_loads.append(W)

    def add_moment(self, M: AppliedMoment): 
        self.moments.append(M)

    # ===== Equivalentes globales =====
    def _equivalents_about(self, x0: float = 0.0) -> Tuple[float, float]:
        """
        Devuelve (sumFy, sumM_x0) de TODAS las cargas externas (sin reacciones),
        tomando momentos respecto de x0 (CCW +).
        """
        sumFy = 0.0
        sumM = 0.0

        # Puntuales
        for p in self.point_loads:
            sumFy += p.magnitude
            sumM += p.magnitude * (p.position - x0)

        # Distribuidas lineales
        for d in self.dist_loads:
            R, xc = d.resultant_and_centroid()
            sumFy += R
            sumM += R * (xc - x0)

        # Momentos aplicados (independientes del punto de momentos)
        for m in self.moments:
            sumM += m.magnitude

        return sumFy, sumM

    # ===== Resolver reacciones según apoyos (versión mejorada) =====
    def solve_reactions(self):
        if self.length <= 0:
            raise ValueError("La longitud de la viga debe ser positiva.")
        
        valid_supports = sorted([s for s in self.supports if 0 <= s.position <= self.length],
                                key=lambda s: s.position)
        
        if not valid_supports:
            valid_supports = [Support(0.0, "pinned"), Support(self.length, "roller")]
            
        incognitas = []
        for s in valid_supports:
            if s.kind == "pinned" or s.kind == "roller":
                incognitas.append(("R", s.position))
            elif s.kind == "fixed":
                incognitas.append(("R", s.position))
                incognitas.append(("M", s.position))
        
        if len(incognitas) != 2:
            raise ValueError(f"Sistema estáticamente indeterminado o inestable. Se esperaban 2 incógnitas, se encontraron {len(incognitas)}.")

        x_react = [pos for _, pos in incognitas]
        
        sumFy_cargas, sumM0_cargas = self._equivalents_about(0.0)

        A = np.zeros((2, 2))
        b = np.zeros(2)

        for i, (kind, x) in enumerate(incognitas):
            if kind == "R":
                A[0, i] = 1.0
                A[1, i] = x
            elif kind == "M":
                A[1, i] = 1.0
        
        b[0] = -sumFy_cargas
        b[1] = -sumM0_cargas
        
        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            raise ValueError("El sistema de ecuaciones no tiene una solución única.")

        self.reactions = {}
        self.fix_moment = {}
        for i, (kind, x) in enumerate(incognitas):
            if kind == "R":
                self.reactions[x] = sol[i]
            elif kind == "M":
                self.fix_moment[x] = sol[i]
        
        # Validación de equilibrio
        sumFy_check = sum(self.reactions.values()) + sumFy_cargas
        sumM0_check = sum(x*R for x,R in self.reactions.items()) + sum(self.fix_moment.values()) + sumM0_cargas
        
        if abs(sumFy_check) > 1e-6 or abs(sumM0_check) > 1e-6:
             raise ValueError("Advertencia: El sistema de reacciones no cumple el equilibrio. Verifique las entradas.")

    # ===== Construcción de w(x), V(x), M(x) (versión original sin cambios) =====
    def generate_arrays(self, n_per_span: int = 200):
        if self.npts <= 1:
            raise ValueError("npts debe ser mayor que 1")

        key = {0.0, self.length}
        key.update([s.position for s in self.supports])
        for d in self.dist_loads:
            key.update([d.x_start, d.x_end])
        key.update([p.position for p in self.point_loads])
        key.update([m.position for m in self.moments])
        xs = sorted(key)

        x_all, w_all, V_all, M_all = [], [], [], []
        V = 0.0
        M = 0.0

        def apply_jumps_at(x0: float):
            nonlocal V, M
            for xr_pos, xr in self.reactions.items():
                if abs(xr_pos - x0) < 1e-12:
                    V += xr
            for p in self.point_loads:
                if abs(p.position - x0) < 1e-12:
                    V += p.magnitude
            for m in self.moments:
                if abs(m.position - x0) < 1e-12:
                    M += m.magnitude
            for xf_pos, mf in self.fix_moment.items():
                if abs(xf_pos - x0) < 1e-12:
                    M += mf

        apply_jumps_at(0.0)

        for a, b in zip(xs[:-1], xs[1:]):
            Lspan = b - a
            n = max(2, int(self.npts * Lspan / self.length))
            x_local = np.linspace(a, b, n)

            w_local = np.zeros_like(x_local)
            for d in self.dist_loads:
                mask = (x_local >= d.x_start) & (x_local <= d.x_end)
                if np.any(mask):
                    Ld = d.x_end - d.x_start
                    if Ld > 1e-12:
                        acoef = (d.w_end - d.w_start) / Ld
                        bcoef = d.w_start - acoef * d.x_start
                        w_local[mask] += acoef * x_local[mask] + bcoef

            for i, x in enumerate(x_local):
                if i > 0:
                    dx = x_local[i] - x_local[i - 1]
                    V += 0.5 * (w_local[i] + w_local[i - 1]) * dx
                    V_prev = V_all[-1] if V_all else V
                    M += 0.5 * (V_prev + V) * dx

                x_all.append(float(x))
                w_all.append(float(w_local[i]))
                V_all.append(float(V))
                M_all.append(float(M))

            apply_jumps_at(b)

        simply_supported = (len(self.reactions) == 2 and len(self.fix_moment) == 0)
        end_moment = any(
            abs(m.position) < 1e-12 or abs(m.position - self.length) < 1e-12
            for m in self.moments
        )
        
        if simply_supported and not end_moment and len(M_all) >= 2:
            L = self.length
            m0 = M_all[0]
            mL = M_all[-1]
            M_all = [M_all[i] - (m0 + (mL - m0) * (x_all[i] / L)) for i in range(len(M_all))]

        self.x = np.array(x_all)
        self.w_field = np.array(w_all)
        self.V = np.array(V_all)
        self.M = np.array(M_all)

        return self.x, self.w_field, self.V, self.M

    # ===== Utilidades de salida =====
    def critical_points_summary(self):
        """
        Devuelve lista de tuplas (x, w, V, M) en
        {0, L} ∪ {x de apoyos} ∪ {x de puntuales} ∪ {extremos de distribuidas} ∪ {x de momentos}.
        """
        xs = {0.0, self.length}
        xs.update([s.position for s in self.supports])
        xs.update([p.position for p in self.point_loads])
        for d in self.dist_loads:
            xs.update([d.x_start, d.x_end])
        xs.update([m.position for m in self.moments])
        xs = sorted(x for x in xs if 0.0 <= x <= self.length)

        if not hasattr(self, "x") or len(self.x) == 0:
            self.generate_arrays()

        def w_exact(xi: float) -> float:
            val = 0.0
            for d in self.dist_loads:
                if d.x_start - 1e-12 <= xi <= d.x_end + 1e-12:
                    Ld = d.x_end - d.x_start
                    if Ld > 1e-12:
                        acoef = (d.w_end - d.w_start) / Ld
                        bcoef = d.w_start - acoef * d.x_start
                        val += acoef * xi + bcoef
            return float(val)

        rows = []
        for xi in xs:
            idx = int(np.argmin(np.abs(self.x - xi)))
            rows.append((float(xi), w_exact(xi), float(self.V[idx]), float(self.M[idx])))
        return rows

    def _coords(self, X, Y, every=4) -> str:
        """Reduce puntos para que el .tex no sea enorme e incluye siempre extremos."""
        X = np.asarray(X); Y = np.asarray(Y)
        n = len(X)
        if n == 0:
            return ""
        if n == 1:
            return f"({X[0]:.6f},{Y[0]:.6f})"
        every = max(1, min(int(every), n - 1))
        idx = np.r_[0, np.arange(every, n - 1, every), n - 1]
        pts = (f"({X[i]:.6f},{Y[i]:.6f})" for i in idx)
        return " ".join(pts)

    def extrema(self):
        if not hasattr(self, "x") or len(self.x) == 0:
            self.generate_arrays()

        def pack(arr):
            arr = np.asarray(arr, dtype=float)
            if len(arr) == 0:
                return {
                    "max_pos": (0.0, 0.0),
                    "max_neg": (0.0, 0.0),
                    "abs_max": (0.0, 0.0),
                }
            i_max = int(np.argmax(arr))
            i_min = int(np.argmin(arr))
            i_abs = int(np.argmax(np.abs(arr)))
            return {
                "max_pos": (float(self.x[i_max]), float(arr[i_max])),
                "max_neg": (float(self.x[i_min]), float(arr[i_min])),
                "abs_max": (float(self.x[i_abs]), float(arr[i_abs])),
            }
        return {"V": pack(self.V), "M": pack(self.M)}

    def _siunitx_unit(self, u: str) -> str:
        u = (u or "").strip().replace("·", ".")
        mapping = {
            "m": r"\metre", "mm": r"\milli\metre", "cm": r"\centi\metre",
            "in": r"\inch", "ft": r"\foot",
            "N": r"\newton", "kN": r"\kilo\newton", "MN": r"\mega\newton", "lb": r"\pound",
            "N/m": r"\newton\per\metre", "kN/m": r"\kilo\newton\per\metre",
            "lb/ft": r"\pound\per\foot",
            "N.m": r"\newton\metre", "kN.m": r"\kilo\newton\metre", "lb.ft": r"\pound\foot",
            "Pa": r"\pascal", "MPa": r"\mega\pascal", "GPa": r"\giga\pascal",
        }
        if u in mapping:
            return mapping[u]
        safe = u.replace("\\", "").replace("{", "").replace("}", "")
        return rf"\text{{{safe}}}"

    def render_latex_document(self) -> str:
        if not hasattr(self, "x") or len(self.x) == 0:
            self.generate_arrays()

        ux, uw, uV, uM = (self.units[k] for k in ("x", "w", "V", "M"))
        ux_tex, uw_tex, uV_tex, uM_tex = (self._siunitx_unit(s) for s in (ux, uw, uV, uM))

        ext = self.extrema()
        xVmax, Vmax = ext["V"]["max_pos"]
        xVmin, Vmin = ext["V"]["max_neg"]
        xVabs, Vabs = ext["V"]["abs_max"]
        xMmax, Mmax = ext["M"]["max_pos"]
        xMmin, Mmin = ext["M"]["max_neg"]
        xMabs, Mabs = ext["M"]["abs_max"]

        sumFy, sumM0 = self._equivalents_about(0.0)
        sumFy_total = sumFy + sum(self.reactions.values())
        sumM0_total = sumM0 + sum(xR * R for xR, R in self.reactions.items()) + sum(self.fix_moment.values())

        if self.supports:
            supports_rows = "\n".join(
                [rf"{s.kind} & \SI{{{s.position:.3f}}}{{{ux_tex}}} \\" for s in self.supports]
            )
        else:
            supports_rows = r"\multicolumn{2}{c}{(sin apoyos)} \\"

        force_rows = [
            rf"\(\mathrm{{R}}(x=\SI{{{xR:.3f}}}{{{ux_tex}}})\) & \SI{{{R:.4f}}}{{{uV_tex}}} \\"
            for xR, R in sorted(self.reactions.items())
        ]
        fix_rows = [
            rf"\(\mathrm{{M}}(x=\SI{{{xM:.3f}}}{{{ux_tex}}})\) & \SI{{{MM:.4f}}}{{{uM_tex}}} \\"
            for xM, MM in sorted(self.fix_moment.items())
        ]
        reactions_rows = "\n".join(force_rows + fix_rows) if (force_rows or fix_rows) else r"\multicolumn{2}{c}{(sin reacciones)} \\"

        fmt = {
            "length": f"{self.length:.3f}",
            "ux_tex": ux_tex,
            "uV_tex": uV_tex,
            "uM_tex": uM_tex,
            "sumFy_total": f"{sumFy_total:.6f}",
            "sumM0_total": f"{sumM0_total:.6f}",
            "Vmax": f"{Vmax:.4f}", "xVmax": f"{xVmax:.3f}",
            "Vmin": f"{Vmin:.4f}", "xVmin": f"{xVmin:.3f}",
            "Vabs": f"{Vabs:.4f}", "xVabs": f"{xVabs:.3f}",
            "Mmax": f"{Mmax:.4f}", "xMmax": f"{xMmax:.3f}",
            "Mmin": f"{Mmin:.4f}", "xMmin": f"{xMmin:.3f}",
            "Mabs": f"{Mabs:.4f}", "xMabs": f"{xMabs:.3f}",
            "supports_rows": supports_rows,
            "reactions_rows": reactions_rows,
        }

        template = Template(r"""
\documentclass[11pt]{article}
\usepackage[spanish]{babel}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{siunitx}
\usepackage{amsmath, amssymb}
\usepackage{geometry}
\geometry{letterpaper, margin=2.3cm}
\sisetup{output-decimal-marker={,}}
\DeclareSIUnit{\inch}{in}
\DeclareSIUnit{\pound}{lb}
\DeclareSIUnit{\foot}{ft}

\begin{document}

\section*{Análisis de viga}

\subsection*{Datos}
Longitud: \(\SI{${length}}{${ux_tex}}\).

\begin{tabular}{l c}
\multicolumn{2}{l}{\textbf{Apoyos}} \\\hline
Tipo & Posición \\\hline
${supports_rows}
\end{tabular}

\medskip

\begin{tabular}{l c}
\multicolumn{2}{l}{\textbf{Reacciones}} \\\hline
Magnitud & Valor \\\hline
${reactions_rows}
\end{tabular}

\medskip

\subsection*{Modelo y ecuaciones}
\[
\frac{dV}{dx} = w(x), \qquad \frac{dM}{dx} = V(x).
\]
Saltos: una fuerza puntual \(P\) o reacción \(R\) produce un salto en \(V\); un momento aplicado \(M_0\) (o de empotramiento) produce un salto en \(M\).

\subsection*{Chequeo de equilibrio}
\[
\Sigma F_y = \SI{${sumFy_total}}{${uV_tex}}, \qquad
\Sigma M_{(x=0)} = \SI{${sumM0_total}}{${uM_tex}}.
\]

\subsection*{Extremos numéricos}
\begin{tabular}{lccc}
 & \textbf{máx (+)} & \textbf{mín (–)} & \(\lvert\cdot\rvert\) \textbf{máx} \\\hline
\(V\) & \(\SI{${Vmax}}{${uV_tex}}\) en \(x=\SI{${xVmax}}{${ux_tex}}\) &
\(\SI{${Vmin}}{${uV_tex}}\) en \(x=\SI{${xVmin}}{${ux_tex}}\) &
\(\SI{${Vabs}}{${uV_tex}}\) en \(x=\SI{${xVabs}}{${ux_tex}}\) \\
\(M\) & \(\SI{${Mmax}}{${uM_tex}}\) en \(x=\SI{${xMmax}}{${ux_tex}}\) &
\(\SI{${Mmin}}{${uM_tex}}\) en \(x=\SI{${xMmin}}{${ux_tex}}\) &
\(\SI{${Mabs}}{${uM_tex}}\) en \(x=\SI{${xMabs}}{${ux_tex}}\) \\
\end{tabular}
\newpage
%\subsection*{Diagramas}
Los diagramas \(V(x)\) y \(M(x)\) se adjuntan a continuación.

\end{document}
""")
        return template.substitute(**fmt)