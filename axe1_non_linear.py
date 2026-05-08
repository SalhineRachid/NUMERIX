# axe1_non_linear.py — Résolution des équations non linéaires
import customtkinter as ctk
import numpy as np
import sympy as sp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import (
    BG, PANEL, CARD, BORDER, ACCENT, ACCENT_DIM, ACCENT_HOV,
    SUCCESS, SUCCESS_DIM, SUCCESS_BRD, DANGER, WARNING,
    T_PRIMARY, T_SECONDARY, T_MUTED,
    C_BLUE, C_AMBER, C_GREEN, C_PURPLE,
    FONT_MAIN, lbl, section_label, entry, primary_btn, separator,
    result_card, rec_card, new_figure, style_ax
)

# ══════════════════════════════════════════════════════════════════════════════
# Numerical Algorithms
# ══════════════════════════════════════════════════════════════════════════════

def dichotomie(f, a, b, tol, max_iter=500):
    """Méthode de la Dichotomie (Bisection)."""
    history = []
    c = a
    for _ in range(max_iter):
        c = (a + b) / 2.0
        history.append(c)
        if abs(b - a) < tol or f(c) == 0:
            break
        if np.sign(f(a)) != np.sign(f(c)):
            b = c
        else:
            a = c
    return c, history


def newton(f, f_prime, x0, epsilon, max_iter=100):
    """Algorithme de Newton-Raphson."""
    x = float(x0)
    history = [x]
    for _ in range(max_iter):
        fp = f_prime(x)
        if abs(fp) < 1e-14:
            raise ValueError("f′(x) ≈ 0 — Newton diverge. Changez le point de départ.")
        x_new = x - f(x) / fp
        history.append(x_new)
        if abs(x_new - x) < epsilon:
            break
        x = x_new
    return x_new, history


def point_fixe(g, x0, epsilon, max_iter=200):
    """Méthode du Point Fixe (x = g(x))."""
    x = float(x0)
    history = [x]
    for _ in range(max_iter):
        x_new = g(x)
        history.append(x_new)
        if abs(x_new - x) < epsilon:
            break
        if abs(x_new) > 1e12:
            raise ValueError("La suite diverge. Choisissez un g(x) contractant.")
        x = x_new
    return x_new, history


# ══════════════════════════════════════════════════════════════════════════════
# UI Frame
# ══════════════════════════════════════════════════════════════════════════════

class Axe1Frame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=BG, corner_radius=0)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()

    # ── Left Input Panel ──────────────────────────────────────────────────────

    def _build_left(self):
        self.left = ctk.CTkScrollableFrame(
            self, width=340, fg_color=PANEL, corner_radius=0,
            scrollbar_button_color="#1c1c1c", scrollbar_button_hover_color="#2a2a2a"
        )
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.grid_columnconfigure(0, weight=1)

        r = 0

        # Header
        ctk.CTkFrame(self.left, height=1, fg_color=BORDER).grid(
            row=r, column=0, sticky="ew"); r += 1

        lbl(self.left, "Équations Non Linéaires", size=15, weight="bold").grid(
            row=r, column=0, padx=20, pady=(22, 2), sticky="w"); r += 1
        lbl(self.left, "Recherche de racines  —  f(x) = 0",
            size=10, color=T_SECONDARY).grid(
            row=r, column=0, padx=20, pady=(0, 18), sticky="w"); r += 1

        # Method selector
        section_label(self.left, "MÉTHODE", r); r += 1
        self.method_var = ctk.StringVar(value="Dichotomie")
        self.seg = ctk.CTkSegmentedButton(
            self.left,
            values=["Dichotomie", "Newton", "Point Fixe"],
            variable=self.method_var,
            fg_color="#0a0a0a",
            selected_color=ACCENT_DIM, selected_hover_color=ACCENT_HOV,
            unselected_color="#0a0a0a", unselected_hover_color="#141414",
            text_color=T_SECONDARY,
            font=ctk.CTkFont(family=FONT_MAIN, size=11),
            command=self._on_method_change
        )
        self.seg.grid(row=r, column=0, padx=20, pady=(0, 18), sticky="ew"); r += 1

        # f(x) entry (shared)
        section_label(self.left, "FONCTION  f(x)", r); r += 1
        self.e_fx = entry(self.left, "ex :  x**3 - x - 2")
        self.e_fx.grid(row=r, column=0, padx=20, pady=(0, 4), sticky="ew"); r += 1
        lbl(self.left, "Syntaxe Python / SymPy  (utilisez  **  pour les puissances)",
            size=9, color="#383838").grid(
            row=r, column=0, padx=20, pady=(0, 16), sticky="w"); r += 1

        # Method-specific fields (dynamic)
        self.dyn_row = r
        self.dyn_frame = ctk.CTkFrame(self.left, fg_color="transparent")
        self.dyn_frame.grid(row=r, column=0, sticky="ew")
        self.dyn_frame.grid_columnconfigure(0, weight=1); r += 1

        # Tolerance (shared)
        section_label(self.left, "TOLÉRANCE", r); r += 1
        self.e_tol = entry(self.left, "1e-7")
        self.e_tol.insert(0, "1e-7")
        self.e_tol.grid(row=r, column=0, padx=20, pady=(0, 18), sticky="ew"); r += 1

        separator(self.left, r); r += 1

        # Run button
        primary_btn(self.left, "▶   Calculer", self._run).grid(
            row=r, column=0, padx=20, pady=(12, 18), sticky="ew"); r += 1

        # Result card
        _, self.res_lbl = result_card(self.left, r); r += 1

        # Recommendation card
        _, self.rec_lbl = rec_card(self.left, r)

        # Initialise dynamic fields
        self._build_dyn_fields()

    def _build_dyn_fields(self):
        for w in self.dyn_frame.winfo_children():
            w.destroy()
        self.dyn_frame.grid_columnconfigure(0, weight=1)
        method = self.method_var.get()

        if method == "Dichotomie":
            section_label(self.dyn_frame, "INTERVALLE  [a, b]", 0)
            ab = ctk.CTkFrame(self.dyn_frame, fg_color="transparent")
            ab.grid(row=1, column=0, padx=20, pady=(0, 16), sticky="ew")
            ab.grid_columnconfigure((0, 1), weight=1)
            self.e_a = entry(ab, "a", width=120)
            self.e_a.grid(row=0, column=0, padx=(0, 6), sticky="ew")
            self.e_b = entry(ab, "b", width=120)
            self.e_b.grid(row=0, column=1, padx=(6, 0), sticky="ew")

        elif method == "Newton":
            section_label(self.dyn_frame, "POINT DE DÉPART  x₀", 0)
            self.e_x0 = entry(self.dyn_frame, "ex :  1.5")
            self.e_x0.grid(row=1, column=0, padx=20, pady=(0, 6), sticky="ew")
            lbl(self.dyn_frame,
                "f′(x) est calculée automatiquement via SymPy.",
                size=9, color="#383838").grid(
                row=2, column=0, padx=20, pady=(0, 16), sticky="w")

        elif method == "Point Fixe":
            section_label(self.dyn_frame, "FONCTION  g(x)  — x = g(x)", 0)
            self.e_gx = entry(self.dyn_frame, "ex :  (x**2 + 2) / (2*x + 1)")
            self.e_gx.grid(row=1, column=0, padx=20, pady=(0, 16), sticky="ew")
            section_label(self.dyn_frame, "POINT DE DÉPART  x₀", 2)
            self.e_x0_pf = entry(self.dyn_frame, "ex :  1.0")
            self.e_x0_pf.grid(row=3, column=0, padx=20, pady=(0, 16), sticky="ew")

    def _on_method_change(self, _=None):
        self._build_dyn_fields()

    # ── Right Plot Panel ──────────────────────────────────────────────────────

    def _build_right(self):
        self.right = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.right.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.right.grid_columnconfigure(0, weight=1)
        self.right.grid_rowconfigure(0, weight=1)

        self.fig, self.ax = new_figure(figsize=(8, 5))
        self.ax.text(0.5, 0.5, "Entrez f(x) et appuyez sur Calculer",
                     ha="center", va="center", color="#2a2a2a", fontsize=13,
                     transform=self.ax.transAxes)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.draw()

    # ── Logic ─────────────────────────────────────────────────────────────────

    def _run(self):
        try:
            func_str = self.e_fx.get().strip()
            if not func_str:
                raise ValueError("Entrez une fonction f(x).")
            tol = float(self.e_tol.get().strip() or "1e-7")

            x_sym = sp.Symbol("x")
            expr   = sp.sympify(func_str, locals={"e": sp.E, "pi": sp.pi})
            f      = sp.lambdify(x_sym, expr, modules=["numpy"])

            method = self.method_var.get()
            root, history = None, []

            if method == "Dichotomie":
                a = float(self.e_a.get())
                b = float(self.e_b.get())
                if f(a) * f(b) >= 0:
                    raise ValueError("f(a) · f(b) ≥ 0 : pas de changement de signe garanti dans [a, b].")
                root, history = dichotomie(f, a, b, tol)
                self._set_recommendation("bisect", expr=expr, a=a, b=b,
                                          niters=len(history))

            elif method == "Newton":
                x0 = float(self.e_x0.get())
                f_prime_expr = sp.diff(expr, x_sym)
                f_prime = sp.lambdify(x_sym, f_prime_expr, modules=["numpy"])
                root, history = newton(f, f_prime, x0, tol)
                self._set_recommendation("newton", expr=expr, fp_expr=f_prime_expr,
                                          x0=x0, niters=len(history))

            elif method == "Point Fixe":
                gx_str  = self.e_gx.get().strip()
                g_expr  = sp.sympify(gx_str, locals={"e": sp.E, "pi": sp.pi})
                g       = sp.lambdify(x_sym, g_expr, modules=["numpy"])
                x0      = float(self.e_x0_pf.get())
                root, history = point_fixe(g, x0, tol)
                self._set_recommendation("fixe", niters=len(history))

            # Display result
            residual = abs(f(root))
            self.res_lbl.configure(
                text=(f"Racine trouvée  :   x* ≈ {root:.10f}\n"
                      f"Résidu  |f(x*)| :   {residual:.3e}\n"
                      f"Itérations      :   {len(history)}\n"
                      f"Méthode         :   {method}"),
                text_color=SUCCESS
            )
            self._plot(f, root, history, func_str, method)

        except Exception as exc:
            self.res_lbl.configure(text=f"⚠  {exc}", text_color=DANGER)

    def _set_recommendation(self, method, **kw):
        msgs = {
            "bisect": (
                f"✓  Dichotomie garantit la convergence si f(a)·f(b) < 0.\n"
                f"   Convergence linéaire : environ {kw.get('niters', '?')} itérations pour votre tolérance.\n"
                f"   Préférez Newton si la dérivée est disponible et non nulle."
            ),
            "newton": (
                f"⚡ Newton — convergence quadratique très rapide ({kw.get('niters', '?')} itérations).\n"
                f"   f′(x) = {kw.get('fp_expr', '?')}\n"
                f"   Attention : peut diverger si f′(x₀) ≈ 0 ou x₀ mal choisi.\n"
                f"   Si divergence → utilisez Dichotomie pour localiser la racine d'abord."
            ),
            "fixe": (
                f"↻  Point Fixe ({kw.get('niters', '?')} itérations).\n"
                f"   Converge si |g′(x)| < 1 au voisinage de la racine.\n"
                f"   Vérifiez que g est bien une contraction sur l'intervalle."
            ),
        }
        self.rec_lbl.configure(text=msgs.get(method, ""))

    def _plot(self, f, root, history, func_str, method):
        self.ax.clear()
        style_ax(self.ax)

        # Compute plot range
        finite_h = [h for h in history if np.isfinite(h)]
        margin   = max(abs(root) * 0.6, 2.5)
        x_min    = min(finite_h + [root]) - margin
        x_max    = max(finite_h + [root]) + margin

        xs = np.linspace(x_min, x_max, 600)
        try:
            ys = np.array(f(xs), dtype=float)
        except Exception:
            ys = np.array([float(f(xi)) for xi in xs])

        # Clip wild values for display
        ys_clip = np.clip(ys, -80, 80)

        # Main curve
        self.ax.plot(xs, ys_clip, color=C_BLUE, linewidth=2.0,
                     label=f"f(x) = {func_str}", zorder=3)
        self.ax.axhline(0, color="#2d2d2d", linewidth=0.8, zorder=1)
        self.ax.axvline(0, color="#2d2d2d", linewidth=0.8, zorder=1)

        # Iteration dots
        if len(history) > 1:
            iter_pts = history[:-1]
            iter_y   = [float(f(xi)) for xi in iter_pts]
            self.ax.scatter(iter_pts, iter_y, color=C_AMBER, s=22, alpha=0.55,
                            zorder=4, label="Itérations")
            # Connect Newton tangent steps
            if method == "Newton" and len(history) >= 2:
                for i in range(min(len(history) - 1, 8)):
                    xi, xi1 = history[i], history[i + 1]
                    self.ax.annotate("", xy=(xi1, 0), xytext=(xi, float(f(xi))),
                                     arrowprops=dict(arrowstyle="-|>",
                                                      color=C_AMBER, alpha=0.4,
                                                      lw=0.8))

        # Root
        self.ax.scatter([root], [0], color=C_GREEN, s=120, zorder=5,
                        label=f"Racine ≈ {root:.7f}", edgecolors="#111111", linewidths=1)

        # Legend & labels
        self.ax.legend(facecolor="#181818", edgecolor="#282828",
                       labelcolor=T_SECONDARY, fontsize=9, framealpha=0.9)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.set_title(f"Visualisation — Méthode {method}", fontsize=11, pad=10)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()
