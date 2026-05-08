#!/usr/bin/env python3
# main.py — NumeriX  ·  Application d'Analyse Numérique
# Génie Logiciel — 3ème Année — Projet Final S5
#
# Démarrage : python main.py
# Dépendances : pip install -r requirements.txt

import matplotlib
matplotlib.use("TkAgg")   # Must be set BEFORE importing pyplot or axe modules

import customtkinter as ctk

# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Import modules AFTER matplotlib backend is set ────────────────────────────
from axe1_non_linear import Axe1Frame
from axe2_linear     import Axe2Frame
from axe3_interpolation import Axe3Frame

# ── Design constants (duplicated from utils for self-contained main) ───────────
BG         = "#111111"
PANEL      = "#0c0c0c"
ACCENT     = "#60a5fa"
ACCENT_DIM = "#1a2a3a"
ACCENT_HOV = "#1e3a5f"
BORDER     = "#1f1f1f"
T_PRIMARY  = "#e2e2e2"
T_SEC      = "#606060"
T_MUTED    = "#2e2e2e"
FONT       = "Helvetica"


class NumeriXApp(ctk.CTk):
    """Root application window."""

    def __init__(self):
        super().__init__()
        self.title("NumeriX — Analyse Numérique")
        self.geometry("1380x840")
        self.minsize(1100, 700)
        self.configure(fg_color=BG)
        self.iconbitmap()   # no icon needed; avoids Tk default icon error on some systems

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()
        self.show("home")

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=248, corner_radius=0,
                                     fg_color=PANEL, border_width=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(8, weight=1)

        # Brand
        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, padx=22, pady=(32, 20), sticky="w")

        ctk.CTkLabel(brand, text="NumeriX",
                     font=ctk.CTkFont(family=FONT, size=28, weight="bold"),
                     text_color=ACCENT).pack(anchor="w")
        ctk.CTkLabel(brand, text="Analyse Numérique  ·  S5",
                     font=ctk.CTkFont(family=FONT, size=10),
                     text_color="#2e2e2e").pack(anchor="w")

        # Separator
        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).grid(
            row=1, column=0, padx=14, pady=(0, 14), sticky="ew")

        # Section label
        ctk.CTkLabel(self.sidebar, text="NAVIGATION",
                     font=ctk.CTkFont(family=FONT, size=9, weight="bold"),
                     text_color="#242424").grid(
            row=2, column=0, padx=22, pady=(0, 6), sticky="w")

        # Nav items: (key, icon, label)
        nav_items = [
            ("home",  "⌂",  "Accueil"),
            ("axe1",  "∿",  "Équations Non Linéaires"),
            ("axe2",  "⊞",  "Systèmes Linéaires"),
            ("axe3",  "◇",  "Interpolation & Approximation"),
        ]
        self.nav_btns: dict[str, ctk.CTkButton] = {}
        for i, (key, icon, label) in enumerate(nav_items):
            btn = ctk.CTkButton(
                self.sidebar,
                text=f"   {icon}  {label}",
                anchor="w",
                font=ctk.CTkFont(family=FONT, size=12),
                height=42, corner_radius=8,
                fg_color="transparent",
                hover_color="#141414",
                text_color="#707070",
                border_width=0,
                command=lambda k=key: self.show(k)
            )
            btn.grid(row=3 + i, column=0, padx=10, pady=2, sticky="ew")
            self.nav_btns[key] = btn

        # Separator before footer
        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).grid(
            row=9, column=0, padx=14, pady=(0, 0), sticky="ew")

        # Footer info
        ctk.CTkLabel(
            self.sidebar,
            text="Génie Logiciel · 3ème Année\n2024 – 2025",
            font=ctk.CTkFont(family=FONT, size=9),
            text_color="#242424",
            justify="left"
        ).grid(row=10, column=0, padx=22, pady=18, sticky="sw")

    # ── Main content area ─────────────────────────────────────────────────────

    def _build_main(self):
        self.container = ctk.CTkFrame(self, corner_radius=0, fg_color=BG)
        self.container.grid(row=0, column=1, sticky="nsew")
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(0, weight=1)

        self.frames: dict[str, ctk.CTkFrame] = {
            "home": self._build_home(),
            "axe1": Axe1Frame(self.container),
            "axe2": Axe2Frame(self.container),
            "axe3": Axe3Frame(self.container),
        }
        for frame in self.frames.values():
            frame.grid(row=0, column=0, sticky="nsew")

    # ── Home Screen ───────────────────────────────────────────────────────────

    def _build_home(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self.container, fg_color=BG, corner_radius=0)
        frame.grid_columnconfigure(0, weight=1)

        # Header bar
        header = ctk.CTkFrame(frame, fg_color=PANEL, corner_radius=0, height=110)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="Bienvenue dans NumeriX",
            font=ctk.CTkFont(family=FONT, size=30, weight="bold"),
            text_color=T_PRIMARY
        ).grid(row=0, column=0, pady=(26, 4))
        ctk.CTkLabel(
            header,
            text="Plateforme d'Analyse Numérique Interactive — Projet Final Génie Logiciel S5",
            font=ctk.CTkFont(family=FONT, size=12),
            text_color="#404040"
        ).grid(row=1, column=0, pady=(0, 18))

        # Thin accent line
        ctk.CTkFrame(frame, height=2, fg_color=ACCENT_DIM).grid(
            row=1, column=0, sticky="ew")

        # Cards row
        cards_outer = ctk.CTkFrame(frame, fg_color="transparent")
        cards_outer.grid(row=2, column=0, pady=50)

        card_data = [
            (
                "axe1", "∿",
                "Équations Non Linéaires",
                "Résolution de f(x) = 0",
                "Dichotomie   ·   Newton   ·   Point Fixe\n"
                "Dérivation symbolique via SymPy\n"
                "Visualisation des itérations",
                ACCENT
            ),
            (
                "axe2", "⊞",
                "Systèmes Linéaires",
                "Résolution de  Ax = b",
                "Gauss (Pivot Partiel)   ·   LU\n"
                "Cholesky   ·   Jacobi   ·   Gauss-Seidel\n"
                "Analyse matricielle & convergence",
                "#a78bfa"
            ),
            (
                "axe3", "◇",
                "Interpolation & Approximation",
                "Ajustement de courbes",
                "Lagrange   ·   Newton (Diff. Divisées)\n"
                "Moindres Carrés Discret & Continu\n"
                "Descente de Gradient",
                "#34d399"
            ),
        ]

        for col, (key, icon, title, subtitle, desc, color) in enumerate(card_data):
            self._make_card(cards_outer, key, icon, title, subtitle, desc, color, col)

        # Bottom caption
        ctk.CTkLabel(
            frame,
            text="Sélectionnez un module dans la barre latérale ou cliquez sur une carte pour commencer.",
            font=ctk.CTkFont(family=FONT, size=11),
            text_color="#2e2e2e"
        ).grid(row=3, column=0, pady=(0, 40))

        return frame

    def _make_card(self, parent, key, icon, title, subtitle, desc, color, col):
        card = ctk.CTkFrame(parent, fg_color="#161616", corner_radius=14,
                             border_width=1, border_color=BORDER,
                             width=295, height=290)
        card.grid(row=0, column=col, padx=14, pady=0)
        card.grid_propagate(False)
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text=icon,
                     font=ctk.CTkFont(family=FONT, size=40),
                     text_color=color).grid(row=0, column=0, pady=(30, 6))

        ctk.CTkLabel(card, text=title,
                     font=ctk.CTkFont(family=FONT, size=14, weight="bold"),
                     text_color=T_PRIMARY, wraplength=260).grid(row=1, column=0, padx=20)

        ctk.CTkLabel(card, text=subtitle,
                     font=ctk.CTkFont(family=FONT, size=10),
                     text_color=color).grid(row=2, column=0, padx=20, pady=(2, 6))

        ctk.CTkLabel(card, text=desc,
                     font=ctk.CTkFont(family=FONT, size=10),
                     text_color="#484848", wraplength=255, justify="center"
                     ).grid(row=3, column=0, padx=20)

        ctk.CTkButton(
            card, text="Ouvrir  →", height=34, corner_radius=8,
            fg_color=ACCENT_DIM, hover_color=ACCENT_HOV, text_color=ACCENT,
            font=ctk.CTkFont(family=FONT, size=11, weight="bold"),
            command=lambda k=key: self.show(k)
        ).grid(row=4, column=0, padx=24, pady=(14, 26), sticky="ew")

    # ── Navigation ────────────────────────────────────────────────────────────

    def show(self, key: str):
        self.frames[key].tkraise()
        for k, btn in self.nav_btns.items():
            if k == key:
                btn.configure(fg_color=ACCENT_DIM, text_color=ACCENT)
            else:
                btn.configure(fg_color="transparent", text_color="#707070")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = NumeriXApp()
    app.mainloop()
