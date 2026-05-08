# axe3_interpolation.py — Interpolation & Approximation
import customtkinter as ctk
import numpy as np
import sympy as sp
from scipy.integrate import quad
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
# Algorithms
# ══════════════════════════════════════════════════════════════════════════════

# ── Lagrange ──────────────────────────────────────────────────────────────────

def lagrange_eval(X, Y, x_vals):
    """Évaluer le polynôme de Lagrange sur x_vals."""
    n      = len(X)
    result = np.zeros_like(x_vals, dtype=float)
    for i in range(n):
        Li = np.ones_like(x_vals, dtype=float)
        for j in range(n):
            if j != i:
                Li *= (x_vals - X[j]) / (X[i] - X[j])
        result += Y[i] * Li
    return result


# ── Newton (Divided Differences) ─────────────────────────────────────────────

def divided_diff_table(X, Y):
    """Build the full divided differences table."""
    n   = len(X)
    tbl = np.zeros((n, n))
    tbl[:, 0] = Y
    for j in range(1, n):
        for i in range(n - j):
            tbl[i, j] = (tbl[i+1, j-1] - tbl[i, j-1]) / (X[i+j] - X[i])
    return tbl


def newton_eval(X, Y, x_vals):
    """Evaluate Newton interpolating polynomial on x_vals."""
    tbl    = divided_diff_table(X, Y)
    coeffs = tbl[0, :]           # diagonal of the table
    n      = len(X)
    result = np.zeros_like(x_vals, dtype=float)
    for k, xv in enumerate(x_vals):
        val  = 0.0
        term = 1.0
        for i in range(n):
            val  += coeffs[i] * term
            term *= (xv - X[i])
        result[k] = val
    return result


# ── Discrete Least Squares (polynomial) ───────────────────────────────────────

def least_squares_discrete(X, Y, degree):
    """Polynomial fit via normal equations (numpy.polyfit)."""
    if degree >= len(X):
        raise ValueError(f"Le degré ({degree}) doit être inférieur au nombre de points ({len(X)}).")
    coeffs = np.polyfit(X, Y, degree)        # highest degree first
    return coeffs


# ── Continuous Least Squares ──────────────────────────────────────────────────

def least_squares_continuous(func_str, a, b, degree):
    """
    Fit f(x) on [a, b] with polynomial of given degree using the Gram matrix.
    G_ij = integral(x^i * x^j, a, b)
    c_i  = integral(x^i * f(x), a, b)
    """
    x_sym = sp.Symbol("x")
    expr  = sp.sympify(func_str, locals={"e": sp.E, "pi": sp.pi})
    f     = sp.lambdify(x_sym, expr, modules=["numpy"])

    m = degree + 1
    G = np.zeros((m, m))
    c = np.zeros(m)

    for i in range(m):
        for j in range(m):
            G[i, j], _ = quad(lambda x: x**(i+j), a, b)
        c[i], _ = quad(lambda x: x**i * f(x), a, b)

    coeffs_low = np.linalg.solve(G, c)        # low-degree first
    return coeffs_low[::-1]                    # return high-degree first (np.poly1d style)


# ── Gradient Descent (linear model y = a·x + b) ───────────────────────────────

def gradient_descent(X, Y, lr=0.01, max_iter=2000, tol=1e-9):
    X, Y = np.array(X, dtype=float), np.array(Y, dtype=float)
    # Normalise learning rate heuristic
    lr_auto = lr / (np.mean(X**2) + 1e-8)
    a, b   = 0.0, 0.0
    n      = len(X)
    hist   = [(a, b)]
    losses = []
    for _ in range(max_iter):
        y_hat = a * X + b
        err   = y_hat - Y
        loss  = float(np.mean(err**2))
        losses.append(loss)
        da    = (2 / n) * float(X @ err)
        db    = (2 / n) * float(np.sum(err))
        a_new = a - lr_auto * da
        b_new = b - lr_auto * db
        if abs(a_new - a) < tol and abs(b_new - b) < tol:
            break
        a, b = a_new, b_new
        hist.append((a, b))
    return a, b, hist, losses


# ══════════════════════════════════════════════════════════════════════════════
# UI Frame
# ══════════════════════════════════════════════════════════════════════════════

class Axe3Frame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=BG, corner_radius=0)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()

    # ── Left Panel ────────────────────────────────────────────────────────────

    def _build_left(self):
        self.left = ctk.CTkScrollableFrame(
            self, width=360, fg_color=PANEL, corner_radius=0,
            scrollbar_button_color="#1c1c1c", scrollbar_button_hover_color="#2a2a2a"
        )
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.grid_columnconfigure(0, weight=1)
        r = 0

        ctk.CTkFrame(self.left, height=1, fg_color=BORDER).grid(
            row=r, column=0, sticky="ew"); r += 1
        lbl(self.left, "Interpolation & Approximation", size=15, weight="bold").grid(
            row=r, column=0, padx=20, pady=(22, 2), sticky="w"); r += 1
        lbl(self.left, "Lagrange · Newton · Moindres Carrés · Descente de Gradient",
            size=10, color=T_SECONDARY).grid(
            row=r, column=0, padx=20, pady=(0, 18), sticky="w"); r += 1

        # Method
        section_label(self.left, "MÉTHODE", r); r += 1
        self.method_var = ctk.StringVar(value="Lagrange")
        methods = [
            "Lagrange", "Newton (Diff. Divisées)",
            "MC Discret (polynomial)", "MC Continu (f(x), [a,b])",
            "Descente de Gradient"
        ]
        self.method_cb = ctk.CTkComboBox(
            self.left, values=methods, variable=self.method_var, width=320,
            fg_color="#191919", border_color="#252525",
            button_color=ACCENT_DIM, button_hover_color=ACCENT_HOV,
            text_color=T_PRIMARY, dropdown_fg_color="#191919",
            dropdown_text_color=T_PRIMARY,
            font=ctk.CTkFont(family=FONT_MAIN, size=12),
            command=self._on_method_change
        )
        self.method_cb.grid(row=r, column=0, padx=20, pady=(0, 16), sticky="ew"); r += 1

        separator(self.left, r); r += 1

        # ── Common point inputs ──
        section_label(self.left, "VALEURS  X  (séparées par des virgules)", r); r += 1
        self.e_X = entry(self.left, "ex :  0, 1, 2, 3, 4")
        self.e_X.grid(row=r, column=0, padx=20, pady=(0, 12), sticky="ew"); r += 1

        section_label(self.left, "VALEURS  Y", r); r += 1
        self.e_Y = entry(self.left, "ex :  0, 1, 4, 9, 16")
        self.e_Y.grid(row=r, column=0, padx=20, pady=(0, 12), sticky="ew"); r += 1

        # ── Dynamic method-specific fields ──
        self.dyn_frame = ctk.CTkFrame(self.left, fg_color="transparent")
        self.dyn_frame.grid(row=r, column=0, sticky="ew")
        self.dyn_frame.grid_columnconfigure(0, weight=1); r += 1

        separator(self.left, r); r += 1

        primary_btn(self.left, "▶   Calculer", self._run).grid(
            row=r, column=0, padx=20, pady=(12, 18), sticky="ew"); r += 1

        _, self.res_lbl = result_card(self.left, r); r += 1
        _, self.rec_lbl = rec_card(self.left, r)

        self._build_dyn_fields()

    def _build_dyn_fields(self):
        for w in self.dyn_frame.winfo_children():
            w.destroy()
        self.dyn_frame.grid_columnconfigure(0, weight=1)
        method = self.method_var.get()

        if method in ("MC Discret (polynomial)", "Descente de Gradient"):
            if method == "MC Discret (polynomial)":
                section_label(self.dyn_frame, "DEGRÉ DU POLYNÔME", 0)
                self.e_deg = entry(self.dyn_frame, "ex :  2")
                self.e_deg.insert(0, "2")
                self.e_deg.grid(row=1, column=0, padx=20, pady=(0, 12), sticky="ew")
            else:
                section_label(self.dyn_frame, "TAUX D'APPRENTISSAGE  α", 0)
                self.e_lr = entry(self.dyn_frame, "ex :  0.01")
                self.e_lr.insert(0, "0.01")
                self.e_lr.grid(row=1, column=0, padx=20, pady=(0, 12), sticky="ew")

        elif method == "MC Continu (f(x), [a,b])":
            lbl(self.dyn_frame, "Ignorez les champs X / Y ci-dessus.",
                size=9, color="#383838").grid(row=0, column=0, padx=20, pady=(0, 4), sticky="w")
            section_label(self.dyn_frame, "FONCTION  f(x)", 1)
            self.e_fx_cont = entry(self.dyn_frame, "ex :  sin(x)")
            self.e_fx_cont.grid(row=2, column=0, padx=20, pady=(0, 12), sticky="ew")
            section_label(self.dyn_frame, "INTERVALLE  [a, b]", 3)
            ab = ctk.CTkFrame(self.dyn_frame, fg_color="transparent")
            ab.grid(row=4, column=0, padx=20, pady=(0, 12), sticky="ew")
            ab.grid_columnconfigure((0, 1), weight=1)
            self.e_a_cont = entry(ab, "a", width=130)
            self.e_a_cont.grid(row=0, column=0, padx=(0, 6), sticky="ew")
            self.e_b_cont = entry(ab, "b", width=130)
            self.e_b_cont.grid(row=0, column=1, padx=(6, 0), sticky="ew")
            section_label(self.dyn_frame, "DEGRÉ DU POLYNÔME", 5)
            self.e_deg_cont = entry(self.dyn_frame, "ex :  3")
            self.e_deg_cont.insert(0, "3")
            self.e_deg_cont.grid(row=6, column=0, padx=20, pady=(0, 12), sticky="ew")

    def _on_method_change(self, _=None):
        self._build_dyn_fields()

    # ── Right Panel ───────────────────────────────────────────────────────────

    def _build_right(self):
        self.right = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.right.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.right.grid_columnconfigure(0, weight=1)
        self.right.grid_rowconfigure(0, weight=1)

        self.fig, self.ax = new_figure(figsize=(8, 5))
        self.ax.text(0.5, 0.5, "Entrez des points et appuyez sur Calculer",
                     ha="center", va="center", color="#2a2a2a", fontsize=12,
                     transform=self.ax.transAxes)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.draw()

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        method = self.method_var.get()
        try:
            if method == "MC Continu (f(x), [a,b])":
                self._run_continuous_ls()
            elif method == "Descente de Gradient":
                self._run_gradient_descent()
            else:
                self._run_point_method(method)
        except Exception as exc:
            self.res_lbl.configure(text=f"⚠  {exc}", text_color=DANGER)

    def _parse_xy(self):
        X = np.array([float(v.strip()) for v in self.e_X.get().split(",") if v.strip()])
        Y = np.array([float(v.strip()) for v in self.e_Y.get().split(",") if v.strip()])
        if len(X) != len(Y):
            raise ValueError("X et Y doivent avoir le même nombre de valeurs.")
        if len(X) < 2:
            raise ValueError("Fournissez au moins 2 points.")
        return X, Y

    def _run_point_method(self, method):
        X, Y = self._parse_xy()
        x_dense = np.linspace(X.min() - 0.5, X.max() + 0.5, 600)

        if method == "Lagrange":
            y_interp = lagrange_eval(X, Y, x_dense)
            label    = "Polynôme de Lagrange"
            tbl_info = ""
            self._set_recommendation("interp", npts=len(X))

        elif method == "Newton (Diff. Divisées)":
            tbl      = divided_diff_table(X, Y)
            y_interp = newton_eval(X, Y, x_dense)
            label    = "Polynôme de Newton"
            # Build table string
            lines = ["Tableau des diff. divisées :"]
            for i in range(len(X)):
                row_str = "  " + "   ".join(f"{tbl[i, j]:+.4f}" for j in range(len(X) - i))
                lines.append(row_str)
            tbl_info = "\n" + "\n".join(lines)
            self._set_recommendation("interp", npts=len(X))

        elif method == "MC Discret (polynomial)":
            deg      = int(self.e_deg.get().strip() or "2")
            coeffs   = least_squares_discrete(X, Y, deg)
            poly     = np.poly1d(coeffs)
            y_interp = poly(x_dense)
            label    = f"MC Discret — degré {deg}"
            tbl_info = f"\nCoefficients (haut→bas) : {np.round(coeffs, 6)}"
            self._set_recommendation("approx", npts=len(X), deg=deg)

        else:
            raise ValueError("Méthode non reconnue.")

        # Residual at data points
        y_at_X  = y_interp[np.searchsorted(x_dense, X)] if False else \
                  np.interp(X, x_dense, y_interp)
        residual = float(np.sqrt(np.mean((y_at_X - Y)**2)))

        self.res_lbl.configure(
            text=(f"Méthode : {label}\n"
                  f"Points  : {len(X)}\n"
                  f"RMSE    : {residual:.6e}" + tbl_info),
            text_color=SUCCESS
        )
        self._plot_interp(X, Y, x_dense, y_interp, label)

    def _run_continuous_ls(self):
        func_str = self.e_fx_cont.get().strip()
        a        = float(self.e_a_cont.get())
        b        = float(self.e_b_cont.get())
        deg      = int(self.e_deg_cont.get().strip() or "3")
        if a >= b:
            raise ValueError("a doit être strictement inférieur à b.")

        coeffs   = least_squares_continuous(func_str, a, b, deg)
        poly     = np.poly1d(coeffs)

        x_sym    = sp.Symbol("x")
        expr     = sp.sympify(func_str, locals={"e": sp.E, "pi": sp.pi})
        f_ref    = sp.lambdify(x_sym, expr, modules=["numpy"])

        x_dense  = np.linspace(a, b, 600)
        y_ref    = f_ref(x_dense)
        y_approx = poly(x_dense)

        self.res_lbl.configure(
            text=(f"MC Continu — f(x)={func_str} sur [{a},{b}]\n"
                  f"Degré : {deg}\n"
                  f"Coefficients (haut→bas) :\n  {np.round(coeffs, 6)}"),
            text_color=SUCCESS
        )
        self._set_recommendation("cont_ls", deg=deg)

        self.ax.clear()
        style_ax(self.ax)
        self.ax.plot(x_dense, y_ref,    color=C_BLUE,   lw=2.0, label=f"f(x) = {func_str}")
        self.ax.plot(x_dense, y_approx, color=C_AMBER,  lw=2.0, linestyle="--",
                     label=f"Approximation (degré {deg})")
        self._finalize_plot("Moindres Carrés Continu")

    def _run_gradient_descent(self):
        X, Y  = self._parse_xy()
        lr    = float(self.e_lr.get().strip() or "0.01")
        a, b, hist, losses = gradient_descent(X, Y, lr=lr)

        x_dense  = np.linspace(X.min() - 0.5, X.max() + 0.5, 400)
        y_line   = a * x_dense + b

        self.res_lbl.configure(
            text=(f"Descente de Gradient\n"
                  f"  y = {a:+.6f} · x  {b:+.6f}\n"
                  f"Itérations  : {len(hist)}\n"
                  f"MSE final   : {losses[-1]:.6e}"),
            text_color=SUCCESS
        )
        self._set_recommendation("gd", niters=len(hist))

        # Two-panel plot: data + line | loss curve
        self.fig.clear()
        ax1, ax2 = self.fig.subplots(1, 2)
        style_ax(ax1); style_ax(ax2)

        ax1.scatter(X, Y, color=C_AMBER, s=60, zorder=4, label="Données")
        ax1.plot(x_dense, y_line, color=C_BLUE, lw=2.0,
                 label=f"y={a:.3f}x{b:+.3f}")
        ax1.set_title("Régression linéaire", fontsize=10, pad=8, color="#808080")
        ax1.legend(facecolor="#181818", edgecolor="#282828",
                   labelcolor=T_SECONDARY, fontsize=8)

        ax2.semilogy(np.arange(len(losses)), losses, color=C_GREEN, lw=1.8)
        ax2.set_title("Courbe de perte (MSE)", fontsize=10, pad=8, color="#808080")
        ax2.set_xlabel("Itération", color=T_SECONDARY)
        ax2.set_ylabel("MSE  (log)", color=T_SECONDARY)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    # ── Recommendation ────────────────────────────────────────────────────────

    def _set_recommendation(self, kind, **kw):
        msgs = {
            "interp": (
                f"✓  Interpolation avec {kw.get('npts','?')} points.\n"
                f"   Le polynôme passe exactement par tous les points.\n"
                f"   Attention aux oscillations de Runge pour n grand.\n"
                f"   Newton est numériquement plus stable que Lagrange."
            ),
            "approx": (
                f"✓  Moindres Carrés discret — degré {kw.get('deg','?')}.\n"
                f"   Ne passe pas nécessairement par les points.\n"
                f"   Idéal pour données expérimentales bruitées.\n"
                f"   Augmentez le degré si l'ajustement est insuffisant."
            ),
            "cont_ls": (
                f"✓  MC Continu : minimise ∫[f(x)−p(x)]² dx sur [a,b].\n"
                f"   Utilise la matrice de Gram et l'intégration numérique.\n"
                f"   Degré {kw.get('deg','?')} utilisé."
            ),
            "gd": (
                f"✓  Gradient Descent : {kw.get('niters','?')} itérations.\n"
                f"   Minimise la MSE par mise à jour : θ ← θ − α·∇MSE.\n"
                f"   Pour polynomial, préférez MC Discret (solution exacte).\n"
                f"   Utile pour grandes données ou modèles non linéaires."
            )
        }
        self.rec_lbl.configure(text=msgs.get(kind, ""))

    # ── Plot helpers ──────────────────────────────────────────────────────────

    def _plot_interp(self, X, Y, x_dense, y_interp, label):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        style_ax(self.ax)
        self.ax.scatter(X, Y, color=C_AMBER, s=70, zorder=5,
                        label="Points de données", edgecolors="#111111", linewidths=1)
        self.ax.plot(x_dense, y_interp, color=C_BLUE, lw=2.0, zorder=3, label=label)
        self._finalize_plot(label)

    def _finalize_plot(self, title=""):
        self.ax.legend(facecolor="#181818", edgecolor="#282828",
                       labelcolor=T_SECONDARY, fontsize=9, framealpha=0.9)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title(title, fontsize=11, pad=10, color="#808080")
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()
