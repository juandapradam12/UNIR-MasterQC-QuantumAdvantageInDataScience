"""Storytelling figures for TFM classification / optimization notebooks.

Saves PNGs under each case folder's ``figures/`` (sibling of ``metrics/``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA

PredictFn = Callable[[np.ndarray], np.ndarray]

UNIR_BLUE = "#0E4A75"
UNIR_CYAN = "#7EC8E3"
UNIR_NAVY = "#062A45"
CUT_EDGE = "#C0392B"
DOM_NODE = "#E67E22"
ROUTE_EDGE = "#C0392B"

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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def _as_numpy(y) -> np.ndarray:
    if hasattr(y, "detach"):
        return np.asarray(y.detach().cpu().numpy())
    if hasattr(y, "to_numpy"):
        return np.asarray(y.to_numpy())
    return np.asarray(y)


# ---------------------------------------------------------------------------
# Classification — Iris
# ---------------------------------------------------------------------------

def plot_iris_pca_story(
    X_train,
    y_train,
    X_test,
    y_test,
    predict_fn: PredictFn,
    figures_dir: Path,
    stem: str,
    title_model: str,
) -> list[Path]:
    """PCA(2) scatter without / with decision regions (via inverse PCA + model)."""
    X_train = _as_numpy(X_train).astype(float)
    X_test = _as_numpy(X_test).astype(float)
    y_train = _as_numpy(y_train).astype(int)
    y_test = _as_numpy(y_test).astype(int)

    pca = PCA(n_components=2, random_state=123)
    pca.fit(X_train)
    tr2 = pca.transform(X_train)
    te2 = pca.transform(X_test)

    classes = np.unique(np.concatenate([y_train, y_test]))
    cmap = plt.get_cmap("tab10")

    # --- without boundary: true labels ---
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=140)
    for c in classes:
        ax.scatter(
            tr2[y_train == c, 0],
            tr2[y_train == c, 1],
            s=36,
            color=cmap(int(c) % 10),
            alpha=0.85,
            label=f"train c{c}",
            edgecolors="none",
        )
        ax.scatter(
            te2[y_test == c, 0],
            te2[y_test == c, 1],
            s=42,
            facecolors="none",
            edgecolors=cmap(int(c) % 10),
            linewidths=1.2,
            label=f"test c{c}",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Iris PCA(2) — etiquetas verdaderas ({title_model})")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    p1 = _savefig(fig, figures_dir / f"{stem}_iris_pca_scatter.png")

    # --- with boundary: mesh predicted via inverse PCA ---
    pad = 0.35
    x_min, x_max = tr2[:, 0].min() - pad, tr2[:, 0].max() + pad
    y_min, y_max = tr2[:, 1].min() - pad, tr2[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid2 = np.c_[xx.ravel(), yy.ravel()]
    grid_x = pca.inverse_transform(grid2)
    z = np.asarray(predict_fn(grid_x)).astype(int).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=140)
    ax.contourf(xx, yy, z, levels=np.arange(z.min(), z.max() + 2) - 0.5, alpha=0.22, cmap="tab10")
    ax.contour(xx, yy, z, levels=np.arange(z.min(), z.max() + 1) + 0.5, colors=UNIR_NAVY, linewidths=0.6, alpha=0.5)
    y_pred_te = np.asarray(predict_fn(X_test)).astype(int)
    for c in classes:
        ax.scatter(
            te2[y_test == c, 0],
            te2[y_test == c, 1],
            s=44,
            color=cmap(int(c) % 10),
            alpha=0.9,
            label=f"true c{c}",
            edgecolors="white",
            linewidths=0.4,
        )
    wrong = y_pred_te != y_test
    if np.any(wrong):
        ax.scatter(
            te2[wrong, 0],
            te2[wrong, 1],
            s=90,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            label="error test",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Iris PCA(2) — frontera {title_model}")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    p2 = _savefig(fig, figures_dir / f"{stem}_iris_pca_boundary.png")
    return [p1, p2]


# ---------------------------------------------------------------------------
# Classification — Bars & Stripes
# ---------------------------------------------------------------------------

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
        ax.set_title(f"y={int(labels[i])}", fontsize=9)
    fig.suptitle(f"Bars and Stripes — muestras ({stem})", y=1.02, fontsize=12)
    fig.tight_layout()
    return _savefig(fig, figures_dir / f"{stem}_bs_samples.png")


def plot_bs_pca_story(
    X_train,
    y_train,
    X_test,
    y_test,
    predict_fn: PredictFn,
    figures_dir: Path,
    stem: str,
    title_model: str,
) -> list[Path]:
    """PCA(2) of model features + decision region (inverse PCA → predict)."""
    # Reuse iris helper naming with bs filenames
    X_train = _as_numpy(X_train).astype(float)
    X_test = _as_numpy(X_test).astype(float)
    y_train = _as_numpy(y_train).astype(int)
    y_test = _as_numpy(y_test).astype(int)

    pca = PCA(n_components=2, random_state=123)
    pca.fit(X_train)
    tr2 = pca.transform(X_train)
    te2 = pca.transform(X_test)
    classes = np.unique(np.concatenate([y_train, y_test]))
    colors = {0: UNIR_CYAN, 1: UNIR_BLUE}

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=140)
    for c in classes:
        ax.scatter(tr2[y_train == c, 0], tr2[y_train == c, 1], s=34, color=colors.get(int(c), cmap_fallback(c)), alpha=0.85, label=f"train y={c}")
        ax.scatter(te2[y_test == c, 0], te2[y_test == c, 1], s=40, facecolors="none", edgecolors=colors.get(int(c), cmap_fallback(c)), linewidths=1.2, label=f"test y={c}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"BS PCA(2) — etiquetas ({title_model})")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    p1 = _savefig(fig, figures_dir / f"{stem}_bs_pca_scatter.png")

    pad = 0.35
    xx, yy = np.meshgrid(
        np.linspace(tr2[:, 0].min() - pad, tr2[:, 0].max() + pad, 200),
        np.linspace(tr2[:, 1].min() - pad, tr2[:, 1].max() + pad, 200),
    )
    grid_x = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    z = np.asarray(predict_fn(grid_x)).astype(int).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=140)
    ax.contourf(xx, yy, z, levels=[-0.5, 0.5, 1.5], colors=[UNIR_CYAN, UNIR_BLUE], alpha=0.25)
    for c in classes:
        ax.scatter(te2[y_test == c, 0], te2[y_test == c, 1], s=44, color=colors.get(int(c), cmap_fallback(c)), edgecolors="white", linewidths=0.4, label=f"test y={c}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"BS PCA(2) — frontera {title_model}")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    p2 = _savefig(fig, figures_dir / f"{stem}_bs_pca_boundary.png")
    return [p1, p2]


def cmap_fallback(c: int):
    return plt.get_cmap("tab10")(int(c) % 10)


# ---------------------------------------------------------------------------
# Classification — Ising
# ---------------------------------------------------------------------------

def _z_expectations_from_amplitudes(X_raw: np.ndarray, n_qubits: int = 4) -> np.ndarray:
    """Reconstruct |ψ⟩ from Re||Im features and return ⟨Z_i⟩ per sample."""
    X_raw = np.asarray(X_raw, dtype=float)
    dim = 2**n_qubits
    if X_raw.shape[1] != 2 * dim:
        # fallback: use first n_qubits PCA-like columns as fake magnetization
        k = min(n_qubits, X_raw.shape[1])
        return X_raw[:, :k]
    re, im = X_raw[:, :dim], X_raw[:, dim:]
    psi = re + 1j * im
    norms = np.linalg.norm(psi, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    psi = psi / norms
    zs = np.zeros((len(psi), n_qubits), dtype=float)
    for q in range(n_qubits):
        # ⟨Z_q⟩ = sum_b (|a_b|^2 * (+1 if bit q==0 else -1))
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
    predict_fn: Optional[PredictFn] = None,
) -> list[Path]:
    """Spin ⟨Z⟩ heatmap + optional PCA boundary on model features."""
    saved: list[Path] = []
    X_train_raw = _as_numpy(X_train_raw).astype(float)
    y_train = _as_numpy(y_train).astype(int)
    zs = _z_expectations_from_amplitudes(X_train_raw, n_qubits=4)

    # sort by label for readability
    order = np.argsort(y_train)
    zs_o = zs[order]
    y_o = y_train[order]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=140)
    im = ax.imshow(zs_o, aspect="auto", cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xlabel("sitio de spin")
    ax.set_ylabel("muestra (ordenada por fase)")
    ax.set_xticks(range(zs.shape[1]))
    ax.set_xticklabels([f"q{i}" for i in range(zs.shape[1])])
    # phase separators
    if len(y_o):
        bounds = np.where(np.diff(y_o) != 0)[0]
        for b in bounds:
            ax.axhline(b + 0.5, color="black", lw=0.8, alpha=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\langle Z_i\rangle$")
    ax.set_title(f"Ising — magnetización local ⟨Z⟩ ({title_model})")
    saved.append(_savefig(fig, figures_dir / f"{stem}_ising_spin_heatmap.png"))

    # chain sketch
    fig, ax = plt.subplots(figsize=(6.0, 1.8), dpi=140)
    G = nx.path_graph(4)
    pos = {i: (i, 0.0) for i in range(4)}
    nx.draw_networkx_nodes(G, pos, node_color=UNIR_BLUE, node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2.0, edge_color=UNIR_NAVY, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: f"q{i}" for i in range(4)}, font_color="white", font_size=10, ax=ax)
    ax.set_axis_off()
    ax.set_title("Ising 1D — cadena de 4 spins (layouts open/closed)")
    saved.append(_savefig(fig, figures_dir / f"{stem}_ising_chain.png"))

    if predict_fn is not None and X_train is not None and X_test is not None and y_test is not None:
        # PCA story on model feature space (already PCA-4 + scaled typically)
        X_train = _as_numpy(X_train).astype(float)
        X_test = _as_numpy(X_test).astype(float)
        y_test = _as_numpy(y_test).astype(int)
        pca = PCA(n_components=2, random_state=123).fit(X_train)
        tr2 = pca.transform(X_train)
        te2 = pca.transform(X_test)
        fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=140)
        for c, name, col in [(0, "ordenado", UNIR_CYAN), (1, "desordenado", UNIR_BLUE)]:
            ax.scatter(tr2[y_train == c, 0], tr2[y_train == c, 1], s=34, color=col, alpha=0.85, label=f"train {name}")
            ax.scatter(te2[y_test == c, 0], te2[y_test == c, 1], s=40, facecolors="none", edgecolors=col, linewidths=1.2, label=f"test {name}")
        pad = 0.35
        xx, yy = np.meshgrid(
            np.linspace(tr2[:, 0].min() - pad, tr2[:, 0].max() + pad, 180),
            np.linspace(tr2[:, 1].min() - pad, tr2[:, 1].max() + pad, 180),
        )
        z = np.asarray(predict_fn(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))).astype(int).reshape(xx.shape)
        ax.contourf(xx, yy, z, levels=[-0.5, 0.5, 1.5], colors=[UNIR_CYAN, UNIR_BLUE], alpha=0.18)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Ising PCA(2) — frontera {title_model}")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)
        saved.append(_savefig(fig, figures_dir / f"{stem}_ising_pca_boundary.png"))
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
    nx.draw_networkx_edges(G, pos, edge_color="#9AA4B2", width=1.5, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"), font_size=9, ax=ax)
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
    nx.draw_networkx_edges(G, pos, edge_color="#D0D5DD", width=1.2, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color=ROUTE_EDGE, width=3.0, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=14)
    ax.set_axis_off()
    ax.set_title(f"TSP — ruta ({title})\n{list(route)}")
    return _savefig(fig, figures_dir / f"{stem}_tsp_route.png")


def plot_mds_instance(nodes, edges, figures_dir: Path, stem: str = "instance") -> Path:
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = {n: MDS_POS.get(n, (hash(n) % 7, hash(n) % 5)) for n in nodes}
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=140)
    nx.draw_networkx_nodes(G, pos, node_color=UNIR_CYAN, node_size=1200, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#9AA4B2", width=1.6, ax=ax)
    ax.set_axis_off()
    ax.set_title("MDS — grafo inicial")
    return _savefig(fig, figures_dir / f"{stem}_mds_graph.png")


def plot_mds_solution(nodes, edges, dominating: Sequence[str], figures_dir: Path, stem: str, title: str) -> Path:
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = {n: MDS_POS.get(n, (0, 0)) for n in nodes}
    dom = set(dominating)
    colors = [DOM_NODE if n in dom else UNIR_CYAN for n in nodes]
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=140)
    nx.draw_networkx_nodes(G, pos, nodelist=list(nodes), node_color=colors, node_size=1200, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#9AA4B2", width=1.6, ax=ax)
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
    nx.draw_networkx_edges(G, pos, edge_color="#9AA4B2", width=1.6, ax=ax)
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
    nx.draw_networkx_edges(G, pos, edgelist=keep_edges, edge_color="#D0D5DD", width=1.2, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color=CUT_EDGE, width=2.8, ax=ax)
    ax.set_axis_off()
    cut_val = len(cut_edges)
    ax.set_title(f"MaxCut — partición ({title})\ncut={cut_val}  bits={bits}")
    return _savefig(fig, figures_dir / f"{stem}_maxcut_cut.png")


def ensure_lib_on_path() -> Path:
    """Insert Notebooks/lib on sys.path; return lib directory."""
    import sys

    for base in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        lib = base / "Notebooks" / "lib"
        if (lib / "storytelling.py").is_file():
            s = str(lib)
            if s not in sys.path:
                sys.path.insert(0, s)
            return lib
    raise FileNotFoundError("Notebooks/lib/storytelling.py not found from cwd parents")
