"""
2D Truss Solver — Matrix Structural Analysis (Direct Stiffness Method)

WHAT THIS SCRIPT NEEDS FROM YOU
===============================

(1) NODE-ELEMENT CONNECTIVITY CHART  (CSV or interactive)
    For every element, supply ONE row:

        element , node1 , node2 , theta_deg , E_GPa , A_mm2 , L_m

    where theta_deg is measured from the +x axis to the vector
    going FROM node1 TO node2 (counter-clockwise positive).

    Notes:
      * Node names are arbitrary strings (A, B, C, 1, 2, ...).
        Each node automatically gets two DOFs: dX and dY.
      * Direction cosines are derived as l = cos(theta), m = sin(theta).
      * Units used internally: E in Pa, A in m^2, L in m  ->  k in N/m.

(2) BOUNDARY CONDITIONS  (one entry per global DOF)
    For every DOF the script asks one of:
        d <value_in_m>   ->  prescribed displacement (use 0 for a support)
        f <value_in_N>   ->  prescribed external force (use 0 for free)
    A DOF must have exactly one of {displacement, force} known.
    If you press <enter>, force = 0 is assumed (free, unloaded).

WHAT YOU GET BACK
=================
    * Each element stiffness matrix k^(e), printed BOTH as
        (A E / L) * [4x4 of direction-cosine pattern]
      and as a numeric matrix with the common power of 10 pulled out.
    * Global stiffness matrix K (assembled, with DOF labels).
    * Solved nodal displacements d.
    * Reaction forces R at constrained DOFs.
    * Axial force F^(e) in every member  (+ tension, - compression).

USAGE
=====
    Interactive:        python truss_solver.py
    From a CSV:         python truss_solver.py connectivity.csv
    With a BC file:     python truss_solver.py connectivity.csv bcs.csv

CSV FORMATS
===========
    connectivity.csv   header row required:
        element,node1,node2,theta_deg,E_GPa,A_mm2,L_m

    bcs.csv            header row required:
        node,dof,d,f
        # dof = x | y
        # d   = prescribed displacement (m), or 'x' if unknown
        # f   = prescribed external force (N), or 'x' if unknown
        # exactly one of d, f must be 'x' for each row
        # any DOF omitted -> defaults to (d=x, f=0): free, unloaded
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Element:
    name: str
    n1: str
    n2: str
    theta_deg: float
    E_Pa: float       # Young's modulus, Pa
    A_m2: float       # cross-section, m^2
    L_m: float        # length, m

    @property
    def l(self) -> float:
        return math.cos(math.radians(self.theta_deg))

    @property
    def m(self) -> float:
        return math.sin(math.radians(self.theta_deg))

    @property
    def AE_over_L(self) -> float:
        return self.A_m2 * self.E_Pa / self.L_m

    def pattern_matrix(self) -> list[list[float]]:
        """Direction-cosine pattern T such that k = (AE/L) * T."""
        l, m = self.l, self.m
        ll, mm, lm = l * l, m * m, l * m
        return [
            [ ll,  lm, -ll, -lm],
            [ lm,  mm, -lm, -mm],
            [-ll, -lm,  ll,  lm],
            [-lm, -mm,  lm,  mm],
        ]

    def k_local(self) -> list[list[float]]:
        c = self.AE_over_L
        return [[c * v for v in row] for row in self.pattern_matrix()]


# ------------------------------------------------------------------
# Pretty-printing
# ------------------------------------------------------------------

def _fmt(x: float, width: int = 10) -> str:
    if abs(x) < 1e-12:
        return f"{0.0:>{width}.4f}"
    return f"{x:>{width}.4f}"


def print_matrix(M, col_labels=None, row_labels=None, title=None, scale=None):
    if title:
        print(title)
    n = len(M)
    cols = len(M[0])
    rlw = max((len(r) for r in (row_labels or [])), default=0)
    cw = 12

    if col_labels:
        header = " " * (rlw + 2)
        for c in col_labels:
            header += f"{c:>{cw}}"
        print(header)

    factor_str = ""
    if scale is not None and scale != 1.0:
        factor_str = f"{scale:.3g} * "
    if factor_str:
        print(f"  [factor: {factor_str}]")

    for i in range(n):
        rl = (row_labels[i] if row_labels else "")
        line = f"{rl:>{rlw}}  "
        for j in range(cols):
            v = M[i][j] / (scale if scale else 1.0)
            line += f"{_fmt(v, cw)}"
        print(line)
    print()


def common_power_of_ten(M) -> float:
    """Return 10**p where p = floor(log10(max|entry|))."""
    mx = max((abs(v) for row in M for v in row), default=0.0)
    if mx == 0.0:
        return 1.0
    p = math.floor(math.log10(mx))
    return 10.0 ** p


# ------------------------------------------------------------------
# Input
# ------------------------------------------------------------------

def read_connectivity_csv(path: str) -> list[Element]:
    out = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(Element(
                name=row["element"].strip(),
                n1=row["node1"].strip(),
                n2=row["node2"].strip(),
                theta_deg=float(row["theta_deg"]),
                E_Pa=float(row["E_GPa"]) * 1e9,
                A_m2=float(row["A_mm2"]) * 1e-6,
                L_m=float(row["L_m"]),
            ))
    return out


def read_connectivity_interactive() -> list[Element]:
    print()
    print("=" * 70)
    print(" NODE-ELEMENT CONNECTIVITY CHART")
    print("=" * 70)
    print(" Enter one element per line.  Blank line to finish.")
    print(" Format:")
    print("    element_name node1 node2 theta_deg E_GPa A_mm2 L_m")
    print(" Example:")
    print("    1 A B 45 200 100 2.828")
    print("-" * 70)
    elements = []
    while True:
        line = input(f" element #{len(elements)+1}: ").strip()
        if not line:
            if not elements:
                print(" Need at least one element.")
                continue
            break
        parts = line.split()
        if len(parts) != 7:
            print(" ! Expected 7 fields, try again.")
            continue
        try:
            elements.append(Element(
                name=parts[0],
                n1=parts[1],
                n2=parts[2],
                theta_deg=float(parts[3]),
                E_Pa=float(parts[4]) * 1e9,
                A_m2=float(parts[5]) * 1e-6,
                L_m=float(parts[6]),
            ))
        except ValueError as e:
            print(f" ! Could not parse ({e}), try again.")
    return elements


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------

def build_dof_index(elements: list[Element]) -> dict[str, int]:
    """Assign DOF indices 2i, 2i+1 in node-first-seen order."""
    nodes: list[str] = []
    seen: set[str] = set()
    for e in elements:
        for n in (e.n1, e.n2):
            if n not in seen:
                seen.add(n)
                nodes.append(n)
    return {n: 2 * i for i, n in enumerate(nodes)}, nodes


def element_dof_map(e: Element, idx: dict[str, int]) -> list[int]:
    a, b = idx[e.n1], idx[e.n2]
    return [a, a + 1, b, b + 1]


def element_dof_labels(e: Element) -> list[str]:
    return [f"d_{e.n1}x", f"d_{e.n1}y", f"d_{e.n2}x", f"d_{e.n2}y"]


def assemble_global(elements, idx, n_nodes):
    n = 2 * n_nodes
    K = [[0.0] * n for _ in range(n)]
    for e in elements:
        ke = e.k_local()
        dmap = element_dof_map(e, idx)
        for i in range(4):
            for j in range(4):
                K[dmap[i]][dmap[j]] += ke[i][j]
    return K


def all_dof_labels(nodes: list[str]) -> list[str]:
    out = []
    for n in nodes:
        out += [f"d_{n}x", f"d_{n}y"]
    return out


def reaction_label(dof_label: str) -> str:
    """d_Ax -> R_Ax  (just swap the leading 'd_' for 'R_')."""
    return "R_" + dof_label[2:]


# ------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------

def _parse_bc_token(tok: str):
    """'x' -> ('x', 0.0)  ;  number -> ('k', float).  Empty -> ('k', 0.0)."""
    s = tok.strip().lower()
    if s == "":
        return "k", 0.0
    if s == "x":
        return "x", 0.0
    return "k", float(s)


def _validate_pair(d_state, f_state, label):
    if d_state == "x" and f_state == "x":
        raise ValueError(
            f"DOF {label}: both displacement and force are unknown — "
            f"need exactly one known."
        )
    if d_state == "k" and f_state == "k":
        raise ValueError(
            f"DOF {label}: both displacement and force are given — "
            f"exactly one must be unknown."
        )


def read_bcs_csv(path, nodes, idx):
    """
    CSV format (header required):
        node,dof,d,f
        # d, f are either a number or 'x' (unknown).
        # Exactly one of d, f must be 'x' for each DOF.
        # Any DOF omitted defaults to: d=x, f=0  (free, unloaded).
    Returns (kind, val) per global DOF, where kind in {'d','f'}:
        'd' -> displacement is known (val = prescribed displacement),
        'f' -> force is known        (val = prescribed external force).
    """
    n = 2 * len(nodes)
    kind = [None] * n
    val = [0.0] * n
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"node", "dof", "d", "f"}
        got = set(reader.fieldnames or [])
        if not required.issubset(got):
            raise ValueError(
                f"BC file {path!r} has columns {sorted(got)}; "
                f"expected at least {sorted(required)}. "
                f"The old 'kind,value' schema is no longer supported — "
                f"use columns node,dof,d,f where d and f are each "
                f"a number or 'x' (unknown)."
            )
        for row in reader:
            node = row["node"].strip()
            ax = row["dof"].strip().lower()
            if node not in idx:
                raise ValueError(f"BC references unknown node {node!r}")
            off = 0 if ax == "x" else 1
            g = idx[node] + off
            ds, dv = _parse_bc_token(row.get("d", ""))
            fs, fv = _parse_bc_token(row.get("f", ""))
            _validate_pair(ds, fs, f"d_{node}{ax}")
            if ds == "k":
                kind[g], val[g] = "d", dv
            else:
                kind[g], val[g] = "f", fv
    # Defaults: any DOF not specified -> free (force = 0).
    for g in range(n):
        if kind[g] is None:
            kind[g] = "f"
            val[g] = 0.0
    return kind, val


def read_bcs_interactive(nodes, idx):
    n = 2 * len(nodes)
    kind = [None] * n
    val = [0.0] * n
    print()
    print("=" * 70)
    print(" BOUNDARY CONDITIONS")
    print("=" * 70)
    print(" For every DOF the script asks TWO questions: its displacement")
    print(" and its force.  EXACTLY ONE of them must be unknown.")
    print("    enter a number (e.g. 0, 500, -1.5e-4)  -> that value is known")
    print("    enter 'x'                              -> that quantity is unknown")
    print(" Unknown forces are reported as reactions (R_<node><axis>);")
    print(" unknown displacements are solved for.")
    print("-" * 70)
    labels = all_dof_labels(nodes)
    for g, lab in enumerate(labels):
        while True:
            try:
                d_in = input(f"  {lab:>8} displacement [m] (number or x): ")
                f_in = input(f"  {lab:>8} force        [N] (number or x): ")
                ds, dv = _parse_bc_token(d_in)
                fs, fv = _parse_bc_token(f_in)
                _validate_pair(ds, fs, lab)
            except ValueError as e:
                print(f"    ! {e}  Try again.")
                continue
            if ds == "k":
                kind[g], val[g] = "d", dv
            else:
                kind[g], val[g] = "f", fv
            break
    return kind, val


# ------------------------------------------------------------------
# Solve  (partition K, solve free DOFs, recover reactions)
# ------------------------------------------------------------------

def _solve_linear(A, b):
    """Gaussian elimination with partial pivoting (no numpy dependency)."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[piv][i]) < 1e-14:
            raise ValueError("Singular reduced stiffness matrix — check BCs.")
        M[i], M[piv] = M[piv], M[i]
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for c in range(i, n + 1):
                M[r][c] -= f * M[i][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def solve(K, kind, val):
    n = len(K)
    free = [i for i in range(n) if kind[i] == "f"]
    fix  = [i for i in range(n) if kind[i] == "d"]
    d = list(val)  # prescribed displacements already in val for 'd' DOFs
    # F at free DOFs is the prescribed external force
    Ff = [val[i] for i in free]
    # Subtract K_fc * d_c
    rhs = [Ff[k] - sum(K[free[k]][c] * d[c] for c in fix) for k in range(len(free))]
    # Reduced K_ff
    Kff = [[K[free[i]][free[j]] for j in range(len(free))] for i in range(len(free))]
    if free:
        df = _solve_linear(Kff, rhs)
        for k, g in enumerate(free):
            d[g] = df[k]
    # Reactions at fixed DOFs: R = K_row * d  -  F_external_at_that_dof (=0 typically)
    R = [0.0] * n
    for i in fix:
        R[i] = sum(K[i][j] * d[j] for j in range(n))
    return d, R, free, fix


def member_forces(elements, idx, d):
    """Axial force F^(e) = (AE/L) * [-l -m  l  m] * d_e."""
    out = []
    for e in elements:
        l, m = e.l, e.m
        dm = element_dof_map(e, idx)
        de = [d[g] for g in dm]
        F = e.AE_over_L * (-l * de[0] - m * de[1] + l * de[2] + m * de[3])
        out.append(F)
    return out


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------

BANNER = """
======================================================================
 2D TRUSS SOLVER  -  Matrix Structural Analysis (Direct Stiffness)
======================================================================
 You will be asked for TWO things:

   1. The Node-Element Connectivity Chart, one row per element:
         element , node1 , node2 , theta_deg , E_GPa , A_mm2 , L_m
      theta_deg = angle from +x axis to the vector node1 -> node2.

   2. Boundary conditions: for every global DOF (each node has dX, dY)
    you give BOTH a displacement and a force, where exactly ONE of
    the two is unknown:
         number    that quantity is known (use 0 for a support / no load)
         x         that quantity is unknown (force x  -> reaction R_<dof>;
                                            disp.  x  -> solved for)

 Internally:  E [Pa], A [m^2], L [m], so stiffness is in N/m.
======================================================================
"""


def main(argv):
    print(BANNER)

    if len(argv) >= 2:
        elements = read_connectivity_csv(argv[1])
        print(f" Loaded {len(elements)} elements from {argv[1]}")
    else:
        elements = read_connectivity_interactive()

    idx, nodes = build_dof_index(elements)

    # ---- Echo parsed elements + detected nodes so wrong-CSV mistakes are
    #      impossible to miss before the BC prompt loop starts.
    print("\n" + "=" * 70)
    print(" CONNECTIVITY LOADED")
    print("=" * 70)
    for e in elements:
        print(f"   el {e.name}:  {e.n1} -> {e.n2},  "
              f"theta = {e.theta_deg:>7.2f} deg,  L = {e.L_m:.4f} m,  "
              f"E = {e.E_Pa/1e9:g} GPa,  A = {e.A_m2*1e6:g} mm^2")
    print(f"\n Detected {len(nodes)} nodes: {', '.join(nodes)}")
    print(f"   -> {2*len(nodes)} global DOFs"
          f"   -> the BC step will collect {2*len(nodes)} (displacement, force) pairs")

    # ---- Global K (assembled first so we can pick ONE shared factor) ----
    K = assemble_global(elements, idx, len(nodes))
    labels = all_dof_labels(nodes)

    # Single common factor used for EVERY element k and the global K
    # so all matrices read on the same scale.
    shared_scale = common_power_of_ten(K)

    # ---- Per-element stiffness matrices ----
    print("\n" + "=" * 70)
    print(" ELEMENT STIFFNESS MATRICES")
    print(f" (every matrix below is shown with the common factor"
          f" {shared_scale:.0e} pulled out)")
    print("=" * 70)
    for e in elements:
        elabels = element_dof_labels(e)
        T = e.pattern_matrix()
        c = e.AE_over_L
        print(f"\n Element {e.name}:  nodes ({e.n1} -> {e.n2}),  "
              f"theta = {e.theta_deg} deg,  l = {e.l:.4f},  m = {e.m:.4f}")
        print(f"   AE/L = ({e.A_m2:.3e} m^2)({e.E_Pa:.3e} Pa)/"
              f"({e.L_m:.4f} m) = {c:.4e} N/m")
        print_matrix(
            T,
            col_labels=elabels, row_labels=elabels,
            title=f"   k^({e.name}) = (AE/L) *",
        )
        ke = e.k_local()
        print_matrix(
            ke,
            col_labels=elabels, row_labels=elabels,
            title=f"   k^({e.name}) = ",
            scale=shared_scale,
        )

    # ---- Global K (display) ----
    print("=" * 70)
    print(" GLOBAL STIFFNESS MATRIX K")
    print("=" * 70)
    print_matrix(K, col_labels=labels, row_labels=labels, title="",
                 scale=shared_scale)

    # ---- Boundary conditions ----
    if len(argv) >= 3:
        kind, val = read_bcs_csv(argv[2], nodes, idx)
        print(f" Loaded BCs from {argv[2]}")
    else:
        kind, val = read_bcs_interactive(nodes, idx)

    print("\n Boundary-condition summary:")
    for g, lab in enumerate(labels):
        if kind[g] == "d":
            unknown = reaction_label(lab)
            print(f"   {lab:>8} = {val[g]:+.6g} m   (known)   "
                  f"-> force {unknown} unknown")
        else:
            print(f"   {lab:>8} unknown   "
                  f"-> force F_{lab[2:]} = {val[g]:+.6g} N (known)")

    # ---- Solve ----
    d, R, free, fix = solve(K, kind, val)

    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    print("\n Nodal displacements:")
    for g, lab in enumerate(labels):
        print(f"   {lab:>8} = {d[g]:+.6e} m")

    print("\n Reaction forces (the unknown forces at constrained DOFs):")
    if not fix:
        print("   (none — no displacement BCs)")
    for g in fix:
        rlab = reaction_label(labels[g])
        print(f"   {rlab:>8} = {R[g]:+.6e} N")

    print("\n Member axial forces  (+ tension, - compression):")
    for e, F in zip(elements, member_forces(elements, idx, d)):
        sign = "T" if F >= 0 else "C"
        print(f"   F^({e.name}) = {F:+.6e} N   ({sign})")

    print()


if __name__ == "__main__":
    main(sys.argv)
