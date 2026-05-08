# utils.py — Shared design tokens and helper factories for NumeriX
import customtkinter as ctk
import matplotlib.pyplot as plt

# ── Design Tokens ─────────────────────────────────────────────────────────────
BG          = "#111111"
PANEL       = "#0c0c0c"
CARD        = "#161616"
BORDER      = "#1f1f1f"
ACCENT      = "#60a5fa"
ACCENT_DIM  = "#1a2a3a"
ACCENT_HOV  = "#1e3a5f"
SUCCESS     = "#4ade80"
SUCCESS_DIM = "#0d1a0d"
SUCCESS_BRD = "#1a3a1a"
WARNING     = "#fbbf24"
DANGER      = "#ef4444"
T_PRIMARY   = "#e2e2e2"
T_SECONDARY = "#606060"
T_MUTED     = "#303030"

FONT_MAIN   = "Helvetica"

# ── Plot Palette ──────────────────────────────────────────────────────────────
C_BLUE   = "#60a5fa"
C_AMBER  = "#f59e0b"
C_GREEN  = "#4ade80"
C_PURPLE = "#a78bfa"
C_CORAL  = "#fb7185"

# ── Helper Factories ──────────────────────────────────────────────────────────

def lbl(parent, text, size=12, weight="normal", color=T_PRIMARY, **kw):
    return ctk.CTkLabel(
        parent, text=text,
        font=ctk.CTkFont(family=FONT_MAIN, size=size, weight=weight),
        text_color=color, **kw
    )

def section_label(parent, text, row, col=0, padx=(20, 20), pady=(4, 4)):
    """Small all-caps section header."""
    lbl(parent, text, size=9, weight="bold", color=T_MUTED).grid(
        row=row, column=col, padx=padx, pady=pady, sticky="w"
    )

def entry(parent, placeholder="", width=300, **kw):
    return ctk.CTkEntry(
        parent, placeholder_text=placeholder, width=width,
        fg_color="#191919", border_color="#252525",
        text_color=T_PRIMARY, placeholder_text_color="#404040",
        font=ctk.CTkFont(family=FONT_MAIN, size=12), **kw
    )

def primary_btn(parent, text, command, **kw):
    return ctk.CTkButton(
        parent, text=text, command=command,
        fg_color=ACCENT_DIM, hover_color=ACCENT_HOV,
        text_color=ACCENT,
        font=ctk.CTkFont(family=FONT_MAIN, size=12, weight="bold"),
        height=38, corner_radius=8, **kw
    )

def separator(parent, row, col=0, padx=16, pady=6):
    ctk.CTkFrame(parent, height=1, fg_color=BORDER).grid(
        row=row, column=col, padx=padx, pady=pady, sticky="ew"
    )

# ── Matplotlib Axes Styling ───────────────────────────────────────────────────

def style_ax(ax):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color("#282828")
    ax.tick_params(colors=T_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(T_SECONDARY)
    ax.yaxis.label.set_color(T_SECONDARY)
    ax.title.set_color("#808080")
    ax.grid(True, alpha=0.07, color="#888888", linestyle="--", linewidth=0.6)

def new_figure(nrows=1, ncols=1, figsize=(8, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(BG)
    if nrows == 1 and ncols == 1:
        style_ax(axes)
    else:
        for ax in (axes.flat if hasattr(axes, 'flat') else [axes]):
            style_ax(ax)
    fig.tight_layout(pad=2.0)
    return fig, axes

# ── Result / Recommendation Cards ────────────────────────────────────────────

def result_card(parent, row):
    """Returns a card frame and its main text label."""
    frame = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=10,
                          border_width=1, border_color=BORDER)
    frame.grid(row=row, column=0, padx=20, pady=(0, 12), sticky="ew")
    frame.grid_columnconfigure(0, weight=1)
    label = ctk.CTkLabel(
        frame, text="Les résultats apparaîtront ici.",
        font=ctk.CTkFont(family=FONT_MAIN, size=11),
        text_color="#404040", wraplength=300, justify="left"
    )
    label.grid(row=0, column=0, padx=16, pady=14, sticky="w")
    return frame, label

def rec_card(parent, row):
    """Returns a green recommendation card frame and its text label."""
    frame = ctk.CTkFrame(parent, fg_color=SUCCESS_DIM, corner_radius=10,
                          border_width=1, border_color=SUCCESS_BRD)
    frame.grid(row=row, column=0, padx=20, pady=(0, 28), sticky="ew")
    frame.grid_columnconfigure(0, weight=1)
    lbl(frame, "💡  Recommandation", size=11, weight="bold", color=SUCCESS).grid(
        row=0, column=0, padx=16, pady=(12, 2), sticky="w"
    )
    text_lbl = ctk.CTkLabel(
        frame, text="Entrez vos données et calculez.",
        font=ctk.CTkFont(family=FONT_MAIN, size=10),
        text_color="#386038", wraplength=300, justify="left"
    )
    text_lbl.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")
    return frame, text_lbl
