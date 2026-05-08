# NumeriX — Application d'Analyse Numérique
### Projet Final · Génie Logiciel 3ème Année · S5

Application desktop interactive regroupant les principaux outils d'analyse numérique
dans une interface dark-mode minimaliste construite avec **CustomTkinter**.

---

## Structure du projet

```
numerix/
├── main.py                  # Point d'entrée — navigation et écran d'accueil
├── axe1_non_linear.py       # Résolution d'équations non linéaires
├── axe2_linear.py           # Résolution de systèmes linéaires
├── axe3_interpolation.py    # Interpolation & Approximation
├── utils.py                 # Design tokens, helpers de widgets, styling matplotlib
├── requirements.txt         # Dépendances Python
└── README.md
```

---

## Dépendances

| Package        | Rôle                                        |
|---------------|---------------------------------------------|
| customtkinter  | Interface graphique dark-mode moderne        |
| numpy          | Calcul numérique (matrices, vecteurs)        |
| scipy          | Décomposition LU, intégration numérique      |
| sympy          | Dérivée symbolique, parsing de f(x)          |
| matplotlib     | Visualisation intégrée (backend TkAgg)       |

---

## Installation

```bash
# 1. Cloner ou décompresser le projet
cd numerix/

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
python main.py
```

---

## Modules & Fonctionnalités

### Axe 1 — Équations Non Linéaires (`axe1_non_linear.py`)

**Algorithmes :**
- **Dichotomie (Bisection)** — convergence linéaire, garantie si f(a)·f(b) < 0
- **Newton-Raphson** — convergence quadratique, f′(x) calculée via SymPy
- **Point Fixe** — converge si |g′(x)| < 1 au voisinage de la racine

**Entrées :** f(x) en syntaxe Python/SymPy (ex: `x**3 - x - 2`), intervalle ou point de départ, tolérance

**Visualisation :** courbe f(x), marqueurs d'itérations, racine mise en évidence

---

### Axe 2 — Systèmes Linéaires (`axe2_linear.py`)

**Méthodes directes :**
- **Gauss** avec pivot partiel — résolution exacte O(n³)
- **LU** (scipy.linalg.lu) — factorisation réutilisable
- **Cholesky** — uniquement pour matrices SPD, 2× plus rapide que LU

**Méthodes itératives :**
- **Jacobi** — converge si la matrice est SDD
- **Gauss-Seidel** — converge ~2× plus vite que Jacobi

**Analyse matricielle :**
- Normes ‖A‖₁, ‖A‖₂, ‖A‖∞
- Rayon spectral ρ(A) = max|λᵢ|
- Détection SDD (Strictly Diagonally Dominant) et SPD

**Visualisation :** graphique en barres de la solution ou courbe de convergence (méthodes itératives)

---

### Axe 3 — Interpolation & Approximation (`axe3_interpolation.py`)

**Interpolation :**
- **Lagrange** — degré n−1, passe par tous les points
- **Newton (Différences Divisées)** — même polynôme, plus stable numériquement ; tableau affiché

**Approximation :**
- **MC Discret (polynomial)** — `numpy.polyfit`, résout les équations normales
- **MC Continu** — matrice de Gram + intégration numérique (scipy.integrate.quad)
- **Descente de Gradient** — modèle linéaire y = ax + b, visualisation de la perte

---

## Système de Recommandation

Chaque axe intègre un panel **💡 Recommandation** qui analyse automatiquement les propriétés
de l'entrée et conseille la méthode la plus adaptée :

| Situation | Recommandation |
|---|---|
| f(a)·f(b) < 0, pas de dérivée | Dichotomie |
| f′(x) connue, bon x₀ | Newton (convergence quadratique) |
| Matrice SPD | Cholesky |
| Matrice SDD et grande taille | Jacobi / Gauss-Seidel |
| Données bruitées | Moindres Carrés plutôt qu'Interpolation |

---

## Saisie des données

**f(x) :** utiliser la syntaxe Python — `x**2`, `sin(x)`, `exp(x)`, `log(x)`, `pi`, `e`

**Matrice :** générez la grille en entrant `n` puis en remplissant les cases. Vecteur `b` en bleu.

**Points (Axe 3) :** entrez X et Y séparés par des virgules — `0, 1, 2, 3, 4`

---

## Captures d'écran

L'interface comprend :
- **Barre latérale** sombre avec navigation icônes
- **Panneaux d'entrée** scrollables à gauche
- **Visualisation matplotlib** intégrée à droite (fond sombre, palette contrastée)
- **Cartes de résultats** et **recommandations** vertes en bas du panneau

---

*Développé avec CustomTkinter 5.x + Matplotlib TkAgg backend*
