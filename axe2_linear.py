# axe2_linear.py — Résolution des systèmes linéaires
import customtkinter as ctk
import numpy as np
from scipy import linalg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import (
    BG, PANEL, CARD, BORDER, ACCENT, ACCENT_DIM, ACCENT_HOV,
    SUCCESS, DANGER, WARNING,
    T_PRIMARY, T_SECONDARY, T_MUTED,
    C_BLUE, C_AMBER, C_GREEN, C_PURPLE, C_CORAL,
    FONT_MAIN, lbl, section_label, entry, primary_btn, separator,
    result_card, rec_card, new_figure, style_ax
)

# ══════════════════════════════════════════════════════════════════════════════
# Matrix Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyse_matrix(A):
    n = A.shape[0]
    norme_1   = float(np.max(np.sum(np.abs(A), axis=0)))
    norme_inf = float(np.max(np.sum(np.abs(A), axis=1)))
    norme_2   = float(np.linalg.norm(A, 2))
    eigenvals = np.linalg.eigvals(A)
    rho       = float(np.max(np.abs(eigenvals)))

    is_sdd = all(
        abs(A[i, i]) > sum(abs(A[i, j]) for j in range(n) if j != i)
        for i in range(n)
    )
    is_sym = bool(np.allclose(A, A.T, atol=1e-10))
    is_spd = False
    if is_sym:
        try:
            ev = np.linalg.eigvalsh(A)
            is_spd = bool(np.all(ev > 1e-14))
        except Exception:
            pass

    return dict(
        norme_1=norme_1, norme_2=norme_2, norme_inf=norme_inf,
        rho=rho, eigenvals=eigenvals,
        is_sdd=is_sdd, is_sym=is_sym, is_spd=is_spd
    )


# ══════════════════════════════════════════════════════════════════════════════
# Direct Methods
# ══════════════════════════════════════════════════════════════════════════════

def gauss_partial_pivot(A, b):
    A, b = A.astype(float).copy(), b.astype(float).copy()
    n = len(b)
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(A[col:, col])))
        A[[col, pivot]] = A[[pivot, col]]
        b[[col, pivot]] = b[[pivot, col]]
        if abs(A[col, col]) < 1e-14:
            raise ValueError("Matrice singulière — pivot nul.")
        for row in range(col + 1, n):
            f = A[row, col] / A[col, col]
            A[row, col:] -= f * A[col, col:]
            b[row]        -= f * b[col]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
    return x, None


def lu_solve(A, b):
    P, L, U = linalg.lu(A)
    Pb = P @ b
    n  = len(b)
    # Forward
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]
    # Backward
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-14:
            raise ValueError("Matrice singulière (U diagonal = 0).")
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x, (L, U, P)


def cholesky_solve(A, b):
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("Cholesky échoue : la matrice n'est pas définie positive.")
    y = linalg.solve_triangular(L,   b, lower=True)
    x = linalg.solve_triangular(L.T, y, lower=False)
    return x, L


# ══════════════════════════════════════════════════════════════════════════════
# Iterative Methods
# ══════════════════════════════════════════════════════════════════════════════

def jacobi(A, b, X0, epsilon, max_iter=500):
    D_inv = np.diag(1.0 / np.diag(A))
    R     = A - np.diag(np.diag(A))
    X     = X0.copy().astype(float)
    hist  = [X.copy()]
    for _ in range(max_iter):
        X_new = D_inv @ (b - R @ X)
        hist.append(X_new.copy())
        if np.linalg.norm(X_new - X, np.inf) < epsilon:
            break
        X = X_new
    return X_new, hist


def gauss_seidel(A, b, X0, epsilon, max_iter=500):
    n = len(b)
    X = X0.copy().astype(float)
    hist = [X.copy()]
    for _ in range(max_iter):
        X_old = X.copy()
        for i in range(n):
            sigma = A[i, :i] @ X[:i] + A[i, i+1:] @ X[i+1:]
            X[i]  = (b[i] - sigma) / A[i, i]
        hist.append(X.copy())
        if np.linalg.norm(X - X_old, np.inf) < epsilon:
            break
    return X, hist


# ══════════════════════════════════════════════════════════════════════════════
# UI Frame
# ══════════════════════════════════════════════════════════════════════════════

class Axe2Frame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=BG, corner_radius=0)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.n = 3
        self._build_left()
        self._build_right()

    # ── Left Panel ────────────────────────────────────────────────────────────

    def _build_left(self):
        self.left = ctk.CTkScrollableFrame(
            self, width=380, fg_color=PANEL, corner_radius=0,
            scrollbar_button_color="#1c1c1c", scrollbar_button_hover_color="#2a2a2a"
        )
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.grid_columnconfigure(0, weight=1)
        r = 0

        ctk.CTkFrame(self.left, height=1, fg_color=BORDER).grid(
            row=r, column=0, sticky="ew"); r += 1
        lbl(self.left, "Systèmes Linéaires", size=15, weight="bold").grid(
            row=r, column=0, padx=20, pady=(22, 2), sticky="w"); r += 1
        lbl(self.left, "Résolution de  Ax = b  — méthodes directes & itératives",
            size=10, color=T_SECONDARY).grid(
            row=r, column=0, padx=20, pady=(0, 18), sticky="w"); r += 1

        # ── Matrix size ──
        section_label(self.left, "TAILLE  n × n", r); r += 1
        sz_row = ctk.CTkFrame(self.left, fg_color="transparent")
        sz_row.grid(row=r, column=0, padx=20, pady=(0, 14), sticky="ew"); r += 1
        sz_row.grid_columnconfigure(0, weight=0)
        sz_row.grid_columnconfigure(1, weight=1)
        self.e_n = ctk.CTkEntry(sz_row, width=70, placeholder_text="n",
                                 fg_color="#191919", border_color="#252525",
                                 text_color=T_PRIMARY,
                                 font=ctk.CTkFont(family=FONT_MAIN, size=12))
        self.e_n.insert(0, "3")
        self.e_n.grid(row=0, column=0, padx=(0, 8))
        ctk.CTkButton(
            sz_row, text="Générer", height=32, corner_radius=8,
            fg_color=ACCENT_DIM, hover_color=ACCENT_HOV, text_color=ACCENT,
            font=ctk.CTkFont(family=FONT_MAIN, size=11),
            command=self._gen_grid
        ).grid(row=0, column=1, sticky="w")

        # ── Matrix grid ──
        section_label(self.left, "MATRICE  A", r); r += 1
        self.mat_container = ctk.CTkFrame(self.left, fg_color="transparent")
        self.mat_container.grid(row=r, column=0, padx=20, pady=(0, 14), sticky="w"); r += 1

        section_label(self.left, "VECTEUR  b", r); r += 1
        self.b_container = ctk.CTkFrame(self.left, fg_color="transparent")
        self.b_container.grid(row=r, column=0, padx=20, pady=(0, 14), sticky="w"); r += 1

        separator(self.left, r); r += 1

        # ── Method ──
        section_label(self.left, "MÉTHODE", r); r += 1
        self.method_var = ctk.StringVar(value="Gauss (Pivot partiel)")
        methods = [
            "Gauss (Pivot partiel)", "Décomposition LU",
            "Cholesky (SPD)", "Jacobi", "Gauss-Seidel"
        ]
        self.method_cb = ctk.CTkComboBox(
            self.left, values=methods, variable=self.method_var, width=340,
            fg_color="#191919", border_color="#252525",
            button_color=ACCENT_DIM, button_hover_color=ACCENT_HOV,
            text_color=T_PRIMARY, dropdown_fg_color="#191919",
            dropdown_text_color=T_PRIMARY,
            font=ctk.CTkFont(family=FONT_MAIN, size=12)
        )
        self.method_cb.grid(row=r, column=0, padx=20, pady=(0, 14), sticky="ew"); r += 1

        # Iterative tolerance
        section_label(self.left, "TOLÉRANCE  (méthodes itératives)", r); r += 1
        self.e_tol = entry(self.left, "1e-8")
        self.e_tol.insert(0, "1e-8")
        self.e_tol.grid(row=r, column=0, padx=20, pady=(0, 14), sticky="ew"); r += 1

        separator(self.left, r); r += 1

        # Run
        primary_btn(self.left, "▶   Résoudre", self._run).grid(
            row=r, column=0, padx=20, pady=(12, 18), sticky="ew"); r += 1

        # Results
        _, self.res_lbl = result_card(self.left, r); r += 1
        _, self.rec_lbl = rec_card(self.left, r)

        # Build initial grid
        self._gen_grid()

    # ── Matrix grid generator ─────────────────────────────────────────────────

    def _gen_grid(self):
        try:
            n = int(self.e_n.get())
            if not (2 <= n <= 9):
                raise ValueError
        except ValueError:
            self.res_lbl.configure(text="⚠  n doit être un entier entre 2 et 9.", text_color=DANGER)
            return
        self.n = n

        for w in self.mat_container.winfo_children():
            w.destroy()
        for w in self.b_container.winfo_children():
            w.destroy()

        cell_w = max(42, min(58, 340 // n))
        self.mat_entries = []
        for i in range(n):
            row_e = []
            for j in range(n):
                e = ctk.CTkEntry(self.mat_container, width=cell_w, height=28,
                                  fg_color="#191919", border_color="#252525",
                                  text_color=T_PRIMARY,
                                  font=ctk.CTkFont(family=FONT_MAIN, size=10))
                e.grid(row=i, column=j, padx=2, pady=2)
                row_e.append(e)
            self.mat_entries.append(row_e)

        self.b_entries = []
        for i in range(n):
            e = ctk.CTkEntry(self.b_container, width=58, height=28,
                              fg_color="#0d1a2d", border_color="#1a3a5f",
                              text_color=ACCENT,
                              font=ctk.CTkFont(family=FONT_MAIN, size=10))
            e.grid(row=i, column=0, padx=2, pady=2)
            self.b_entries.append(e)

    def _read_matrix(self):
        n = self.n
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i in range(n):
            for j in range(n):
                v = self.mat_entries[i][j].get().strip()
                A[i, j] = float(v) if v else 0.0
        for i in range(n):
            v = self.b_entries[i].get().strip()
            b[i] = float(v) if v else 0.0
        return A, b

    # ── Right Panel ───────────────────────────────────────────────────────────

    def _build_right(self):
        self.right = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.right.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.right.grid_columnconfigure(0, weight=1)
        self.right.grid_rowconfigure(0, weight=1)

        self.fig, self.ax = new_figure(figsize=(8, 5))
        self.ax.text(0.5, 0.5, "Entrez une matrice et appuyez sur Résoudre",
                     ha="center", va="center", color="#2a2a2a", fontsize=12,
                     transform=self.ax.transAxes)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.draw()

    # ── Solve ─────────────────────────────────────────────────────────────────

    def _run(self):
        try:
            A, b = self._read_matrix()
        except ValueError as exc:
            self.res_lbl.configure(text=f"⚠  Erreur de lecture : {exc}", text_color=DANGER)
            return

        try:
            props   = analyse_matrix(A)
            method  = self.method_var.get()
            tol     = float(self.e_tol.get().strip() or "1e-8")
            hist    = None
            aux     = None

            if method == "Gauss (Pivot partiel)":
                x, _ = gauss_partial_pivot(A, b)
                info  = "Gauss avec pivot partiel — solution exacte"

            elif method == "Décomposition LU":
                x, aux = lu_solve(A, b)
                info = "Décomposition LU (scipy.linalg.lu) — solution exacte"

            elif method == "Cholesky (SPD)":
                if not props["is_spd"]:
                    raise ValueError(
                        "La matrice n'est pas symétrique définie positive.\n"
                        "Cholesky ne peut pas être appliqué.")
                x, aux = cholesky_solve(A, b)
                info = "Décomposition de Cholesky — matrice SPD ✓"

            elif method == "Jacobi":
                x, hist = jacobi(A, b, np.zeros(self.n), tol)
                info = f"Méthode de Jacobi — {len(hist)} itérations"

            elif method == "Gauss-Seidel":
                x, hist = gauss_seidel(A, b, np.zeros(self.n), tol)
                info = f"Méthode de Gauss-Seidel — {len(hist)} itérations"

            else:
                raise ValueError("Méthode inconnue.")

            residual = float(np.linalg.norm(A @ x - b))
            sol_str  = "\n".join(f"  x{i+1} = {xi:+.10f}" for i, xi in enumerate(x))
            self.res_lbl.configure(
                text=(f"Solution x :\n{sol_str}\n\n"
                      f"‖Ax − b‖₂  :  {residual:.3e}\n"
                      f"Info        :  {info}"),
                text_color=SUCCESS
            )
            self._set_recommendation(method, props, hist)
            self._plot(A, b, x, hist, props, method)

        except Exception as exc:
            self.res_lbl.configure(text=f"⚠  {exc}", text_color=DANGER)

    # ── Recommendation ────────────────────────────────────────────────────────

    def _set_recommendation(self, method, props, hist):
        lines = []
        if props["is_spd"]:
            lines.append("✓  Matrice SPD détectée → Cholesky est 2× plus rapide que LU.")
        if props["is_sdd"]:
            lines.append("✓  Matrice SDD → convergence garantie (Jacobi & Gauss-Seidel).")
        else:
            lines.append("⚠  Matrice non-SDD → préférez Gauss ou LU pour fiabilité.")
        lines.append(f"   ρ(A) = {props['rho']:.4f} {'< 1 → itératifs convergent.' if props['rho'] < 1 else '≥ 1 → convergence itérative non garantie.'}")
        if hist:
            lines.append(f"   Gauss-Seidel converge généralement 2× plus vite que Jacobi.")
        self.rec_lbl.configure(text="\n".join(lines))

    # ── Plot ──────────────────────────────────────────────────────────────────

    def _plot(self, A, b, x, hist, props, method):
        self.ax.clear()
        style_ax(self.ax)

        if hist and len(hist) > 2:
            # Convergence curve
            errors = [np.linalg.norm(h - x, np.inf) for h in hist[:-1]]
            # Filter zeros for log scale
            errors = [max(e, 1e-16) for e in errors]
            iters  = np.arange(len(errors))
            self.ax.semilogy(iters, errors, color=C_BLUE, linewidth=2,
                             label="‖x_k − x*‖∞")
            self.ax.scatter(iters, errors, color=C_AMBER, s=18, alpha=0.7, zorder=4)
            self.ax.set_xlabel("Itération")
            self.ax.set_ylabel("‖x_k − x*‖∞  (log)")
            self.ax.set_title(f"Convergence — {method}", fontsize=11, pad=10)
            self.ax.legend(facecolor="#181818", edgecolor="#282828",
                           labelcolor=T_SECONDARY, fontsize=9)
        else:
            # Bar chart of solution components
            labels = [f"x{i+1}" for i in range(len(x))]
            colors = [C_BLUE if xi >= 0 else C_CORAL for xi in x]
            bars   = self.ax.bar(labels, x, color=colors, alpha=0.82,
                                  edgecolor="#222222", linewidth=0.8)
            self.ax.axhline(0, color="#2d2d2d", linewidth=0.8)
            for bar, val in zip(bars, x):
                self.ax.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + (max(abs(x)) * 0.02 if val >= 0 else -max(abs(x)) * 0.06),
                              f"{val:.4f}", ha="center", va="bottom",
                              color=T_SECONDARY, fontsize=8)
            self.ax.set_xlabel("Variable")
            self.ax.set_ylabel("Valeur")
            self.ax.set_title(f"Solution  Ax = b  — {method}", fontsize=11, pad=10)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()
