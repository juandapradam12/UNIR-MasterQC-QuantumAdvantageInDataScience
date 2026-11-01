"""Storytelling figures for TFM classification / optimization notebooks.

Saves PNGs under each case folder's ``figures/`` and shows them inline
(``plt.show()``). Classification plots reuse predictions already computed
in the notebook — they never call ``model.predict`` again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

# Paleta UNIR: azules + grises (misma familia que Results_*)
UNIR_NAVY = "#0B1F33"
UNIR_BLUE = "#0E4A75"
UNIR_AZUL = "#219BC3"
UNIR_CYAN = "#7EC8E3"
UNIR_ICE = "#A8D5E5"
UNIR_MIST = "#D6EEF5"
UNIR_GRIS = "#858591"
UNIR_CLARO = "#C0C0C8"
UNIR_LIGHT = "#D0D5DD"

# Clases / acentos dentro de la paleta (sin naranjas/rojos/tab10)
CLASS_COLORS = [UNIR_NAVY, UNIR_AZUL, UNIR_CYAN, UNIR_GRIS, UNIR_ICE, UNIR_BLUE]
BINARY_COLORS = {0: UNIR_CYAN, 1: UNIR_BLUE}
HIGHLIGHT = UNIR_NAVY  # rutas / cut / dominating set
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "unir_bluegray", [UNIR_MIST, UNIR_CYAN, UNIR_BLUE, UNIR_NAVY]
)

TSP_POS = {
    0: (0.0, 0.0),
    1: (1.0, 1.0),
    2: (2.0, 0.0),
    3: (1.0, -1.0),
    4: (-1.0, -1.0),
}

MDS_POS = {
    "Mario": (0.0, 1.0),
    "Sarah": (-1.0, 0.3),
    "Raul": (1.0, 0.3),
    "Ana": (-0.3, -0.8),
    "Enrique": (-1.6, -0.6),
    "Saul": (1.2, -0.9),
}


def resolve_figures_dir(family: str, case: str) -> Path:
    """Resolve ``Notebooks/{family}/{case}/figures`` from cwd or parents."""
    for base in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        case_dir = base / "Notebooks" / family / case
        if case_dir.is_dir():
            fig_dir = case_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            return fig_dir
    fig_dir = Path.cwd().resolve() / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _savefig(fig: plt.Figure, path: Path) -> Path:
    """Save PNG and show inline (do not close before show)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=150)
    print(f"Saved figure: {path}")
    plt.show()
    return path


def _as_numpy(y) -> np.ndarray:
    if hasattr(y, "detach"):
        return np.asarray(y.detach().cpu().numpy())
    if hasattr(y, "to_numpy"):
        return np.asarray(y.to_numpy())
    return np.asarray(y)


def _class_color(c: int) -> str:
    return CLASS_COLORS[int(c) % len(CLASS_COLORS)]


def _style_axes(ax):
    ax.grid(True, alpha=0.25, color=UNIR_CLARO)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(UNIR_CLARO)
    ax.spines["bottom"].set_color(UNIR_CLARO)


# ---------------------------------------------------------------------------
# Classification — Iris / BS / Ising PCA stories (reuse notebook predictions)
# ---------------------------------------------------------------------------

def _pca2(X_train, X_test):
    X_train = _as_numpy(X_train).astype(float)
    X_test = _as_numpy(X_test).astype(float)
    pca = PCA(n_components=2, random_state=123)
    tr2 = pca.fit_transform(X_train)
    te2 = pca.transform(X_test)
    return tr2, te2


def plot_pca_true_and_pred(
    X_train,
    y_train,
    X_test,
    y_test,
    y_pred_test,
    figures_dir: Path,
    stem: str,
    title_model: str,
    dataset_tag: str,
) -> list[Path]:
    """PCA(2): (1) etiquetas verdaderas (2) predicciones ya evaluadas en el notebook.

    No llama al modelo: solo proyecta y pinta ``y_pred_test``.
    """
    y_train = _as_numpy(y_train).astype(int)
    y_test = _as_numpy(y_test).astype(int)
    y_pred_test = _as_numpy(y_pred_test).astype(int)
    tr2, te2 = _pca2(X_train, X_test)
    classes = np.unique(np.concatenate([y_train, y_test]))

    # --- sin frontera: etiquetas verdaderas ---
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=140)
    for c in classes:
        col = _class_color(c)
        ax.scatter(
            tr2[y_train == c, 0],
            tr2[y_train == c, 1],
            s=36,
            color=col,
            alpha=0.85,
            label=f"train c{c}",
            edgecolors="none",
        )
        ax.scatter(
            te2[y_test == c, 0],
            te2[y_test == c, 1],
            s=42,
            facecolors="none",
            edgecolors=col,
            linewidths=1.2,
            label=f"test c{c}",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{dataset_tag} PCA(2) — etiquetas verdaderas ({title_model})")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    _style_axes(ax)
    p1 = _savefig(fig, figures_dir / f"{stem}_{dataset_tag.lower()}_pca_scatter.png")

    # --- con resultado del clasificador: predicciones existentes ---
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=140)
    for c in classes:
        col = _class_color(c)
        mask = y_pred_test == c
        ax.scatter(
            te2[mask, 0],
            te2[mask, 1],
            s=48,
            color=col,
            alpha=0.9,
            label=f"pred c{c}",
            edgecolors="white",
            linewidths=0.4,
        )
    wrong = y_pred_test != y_test
    if np.any(wrong):
        ax.scatter(
            te2[wrong, 0],
            te2[wrong, 1],
            s=100,
            facecolors="none",
            edgecolors=UNIR_NAVY,
            linewidths=1.6,
            label="error test",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{dataset_tag} PCA(2) — predicciones test ({title_model})")
    ax.legend(fontsize=8, frameon=False)
    _style_axes(ax)
    p2 = _savefig(fig, figures_dir / f"{stem}_{dataset_tag.lower()}_pca_pred.png")
    return [p1, p2]


def plot_iris_pca_story(
    X_train,
    y_train,
    X_test,
    y_test,
    y_pred_test,
    figures_dir: Path,
    stem: str,
    title_model: str,
) -> list[Path]:
    return plot_pca_true_and_pred(
        X_train, y_train, X_test, y_test, y_pred_test,
        figures_dir, stem, title_model, dataset_tag="Iris",
    )


def plot_bs_samples(
    images: np.ndarray,
    labels: np.ndarray,
    figures_dir: Path,
    stem: str,
    n_show: int = 8,
) -> Path:
    """Grid of Bars and Stripes patterns."""
    images = np.asarray(images)
    labels = _as_numpy(labels).astype(int)
    n_show = min(n_show, len(images))
    cols = 4
    rows = int(np.ceil(n_show / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(9.5, 2.4 * rows), dpi=140)
    axs = np.atleast_2d(axs)
    for i in range(rows * cols):
        ax = axs[i // cols, i % cols]
        ax.axis("off")
        if i >= n_show:
            continue
        img = images[i].reshape(4, 4)
        ax.imshow(img, cmap="gray", vmin=float(img.min()), vmax=float(img.max()))
        ax.set_title(f"y={int(labels[i])}", fontsize=9, color=UNIR_BLUE)
    fig.suptitle(f"Bars and Stripes — muestras ({stem})", y=1.02, fontsize=12, color=UNIR_NAVY)
    fig.tight_layout()
    return _savefig(fig, figures_dir / f"{stem}_bs_samples.png")


def plot_bs_pca_story(
    X_train,
    y_train,
    X_test,
    y_test,
    y_pred_test,
    figures_dir: Path,
    stem: str,
    title_model: str,
) -> list[Path]:
    return plot_pca_true_and_pred(
        X_train, y_train, X_test, y_test, y_pred_test,
        figures_dir, stem, title_model, dataset_tag="BS",
    )


def _z_expectations_from_amplitudes(X_raw: np.ndarray, n_qubits: int = 4) -> np.ndarray:
    """Reconstruct |ψ⟩ from Re||Im features and return ⟨Z_i⟩ per sample."""
    X_raw = np.asarray(X_raw, dtype=float)
    dim = 2**n_qubits
    if X_raw.shape[1] != 2 * dim:
        k = min(n_qubits, X_raw.shape[1])
        return X_raw[:, :k]
    re, im = X_raw[:, :dim], X_raw[:, dim:]
    psi = re + 1j * im
    norms = np.linalg.norm(psi, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    psi = psi / norms
    zs = np.zeros((len(psi), n_qubits), dtype=float)
    for q in range(n_qubits):
        for b in range(dim):
            bit = (b >> q) & 1
            sign = 1.0 if bit == 0 else -1.0
            zs[:, q] += sign * (np.abs(psi[:, b]) ** 2)
    return zs


def plot_ising_story(
    X_train_raw,
    y_train,
    figures_dir: Path,
    stem: str,
    title_model: str,
    X_train=None,
    X_test=None,
    y_test=None,
    y_pred_test=None,
) -> list[Path]:
    """Spin ⟨Z⟩ heatmap + chain + optional PCA true/pred (reuses notebook preds)."""
    saved: list[Path] = []
    X_train_raw = _as_numpy(X_train_raw).astype(float)
    y_train = _as_numpy(y_train).astype(int)
    zs = _z_expectations_from_amplitudes(X_train_raw, n_qubits=4)

    order = np.argsort(y_train)
    zs_o = zs[order]
    y_o = y_train[order]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=140)
    im = ax.imshow(zs_o, aspect="auto", cmap=HEAT_CMAP, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xlabel("sitio de spin")
    ax.set_ylabel("muestra (ordenada por fase)")
    ax.set_xticks(range(zs.shape[1]))
    ax.set_xticklabels([f"q{i}" for i in range(zs.shape[1])])
    if len(y_o):
        for b in np.where(np.diff(y_o) != 0)[0]:
            ax.axhline(b + 0.5, color=UNIR_NAVY, lw=0.8, alpha=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\langle Z_i\rangle$")
    ax.set_title(f"Ising — magnetización local ⟨Z⟩ ({title_model})")
    saved.append(_savefig(fig, figures_dir / f"{stem}_ising_spin_heatmap.png"))

    fig, ax = plt.subplots(figsize=(6.0, 1.8), dpi=140)
    G = nx.path_graph(4)
    pos = {i: (i, 0.0) for i in range(4)}
    nx.draw_networkx_nodes(G, pos, node_color=UNIR_BLUE, node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2.0, edge_color=UNIR_GRIS, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: f"q{i}" for i in range(4)}, font_color="white", font_size=10, ax=ax)
    ax.set_axis_off()
    ax.set_title("Ising 1D — cadena de 4 spins (layouts open/closed)")
    saved.append(_savefig(fig, figures_dir / f"{stem}_ising_chain.png"))

    if (
        X_train is not None
        and X_test is not None
        and y_test is not None
        and y_pred_test is not None
    ):
        saved.extend(
            plot_pca_true_and_pred(
                X_train, y_train, X_test, y_test, y_pred_test,
                figures_dir, stem, title_model, dataset_tag="Ising",
            )
        )
    return saved


# ---------------------------------------------------------------------------
# Optimization — TSP / MDS / MaxCut
# ---------------------------------------------------------------------------

def plot_tsp_instance(nodes, dist: dict, figures_dir: Path, stem: str = "instance") -> Path:
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (i, j), w in dist.items():
        if i < j:
            G.add_edge(i, j, weight=w)
    pos = {k: TSP_POS[k] for k in nodes}
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=140)
    nx.draw_networkx_nodes(G, pos, node_color=UNIR_BLUE, node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=12, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=UNIR_CLARO, width=1.5, ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=nx.get_edge_attributes(G, "weight"), font_size=9, ax=ax, font_color=UNIR_GRIS
    )
    ax.set_axis_off()
    ax.set_title("TSP — grafo inicial (distancias)")
    return _savefig(fig, figures_dir / f"{stem}_tsp_graph.png")


def plot_tsp_route(nodes, dist: dict, route: Sequence[int], figures_dir: Path, stem: str, title: str) -> Path:
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (i, j), w in dist.items():
        if i < j:
            G.add_edge(i, j, weight=w)
    pos = {k: TSP_POS[k] for k in nodes}
    route_edges = list(zip(route[:-1], route[1:]))
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=140)
    nx.draw_networkx_nodes(G, pos, node_color=UNIR_BLUE, node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=12, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=UNIR_LIGHT, width=1.2, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=route_edges, edge_color=HIGHLIGHT, width=3.0, ax=ax,
        arrows=True, arrowstyle="-|>", arrowsize=14,
    )
    ax.set_axis_off()
    ax.set_title(f"TSP — ruta ({title})\n{list(route)}")
    return _savefig(fig, figures_dir / f"{stem}_tsp_route.png")


def plot_mds_instance(nodes, edges, figures_dir: Path, stem: str = "instance") -> Path:
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = {n: MDS_POS.get(n, (0, 0)) for n in nodes}
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=140)
    nx.draw_networkx_nodes(G, pos, node_color=UNIR_CYAN, node_size=1200, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax, font_color=UNIR_NAVY)
    nx.draw_networkx_edges(G, pos, edge_color=UNIR_CLARO, width=1.6, ax=ax)
    ax.set_axis_off()
    ax.set_title("MDS — grafo inicial")
    return _savefig(fig, figures_dir / f"{stem}_mds_graph.png")


def plot_mds_solution(nodes, edges, dominating: Sequence[str], figures_dir: Path, stem: str, title: str) -> Path:
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = {n: MDS_POS.get(n, (0, 0)) for n in nodes}
    dom = set(dominating)
    colors = [HIGHLIGHT if n in dom else UNIR_CYAN for n in nodes]
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=140)
    nx.draw_networkx_nodes(G, pos, nodelist=list(nodes), node_color=colors, node_size=1200, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax, font_color="white")
    nx.draw_networkx_edges(G, pos, edge_color=UNIR_CLARO, width=1.6, ax=ax)
    ax.set_axis_off()
    ax.set_title(f"MDS — dominating set ({title})\n{list(dominating)}")
    return _savefig(fig, figures_dir / f"{stem}_mds_dominating_set.png")


def plot_maxcut_instance(n: int, edges, figures_dir: Path, stem: str = "instance") -> Path:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=140)
    nx.draw_networkx_nodes(G, pos, node_color=UNIR_BLUE, node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=11, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=UNIR_CLARO, width=1.6, ax=ax)
    ax.set_axis_off()
    ax.set_title(f"MaxCut — grafo inicial (n={n}, |E|={len(list(edges))})")
    return _savefig(fig, figures_dir / f"{stem}_maxcut_graph.png")


def plot_maxcut_partition(n: int, edges, bits: Sequence[int], figures_dir: Path, stem: str, title: str) -> Path:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = nx.circular_layout(G)
    bits = list(bits)
    node_colors = [UNIR_BLUE if b else UNIR_CYAN for b in bits]
    cut_edges = [(u, v) for u, v in G.edges() if bits[u] != bits[v]]
    keep_edges = [(u, v) for u, v in G.edges() if bits[u] == bits[v]]
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=140)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=11, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=keep_edges, edge_color=UNIR_LIGHT, width=1.2, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color=HIGHLIGHT, width=2.8, ax=ax)
    ax.set_axis_off()
    ax.set_title(f"MaxCut — partición ({title})\ncut={len(cut_edges)}  bits={bits}")
    return _savefig(fig, figures_dir / f"{stem}_maxcut_cut.png")
