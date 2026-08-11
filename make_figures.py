#!/usr/bin/env python3
"""
make_figures.py
===============
Gera as figuras do artigo a partir dos artefatos ja produzidos.

ENTRADAS ESPERADAS (no diretorio atual, salvo indicacao)
    manifest_3w.parquet        passo 1
    taxonomy_pairs.parquet     passo 4
    class9_viability.parquet   passo 5
    windows_class9.npz         passo 7
    results.parquet            passo 8  (REEXECUTAR com make_folds corrigido)
    results_seeds.parquet      passo 9
    interpolation_test.parquet passo 10

SAIDA
    figs/fig1_well_class.pdf        matriz poco x classe
    figs/fig2_availability.pdf      disponibilidade de variaveis por dominio
    figs/fig3_missingness.pdf       binaridade dos faltantes
    figs/fig4_taxonomy.pdf          taxonomia de sinal por variavel
    figs/fig5_leakage.pdf           orcamento de vazamento
    figs/fig6_folds.pdf             AUROC por dobra sob LOWO
    figs/fig7_interpolation.pdf     interpolacao: real vs simulado

Uso:
    python make_figures.py [--raiz=CAMINHO] [--out=figs]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LogNorm

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

SHARED = ["P-MON-CKP", "P-PDG", "P-TPT", "T-JUS-CKP", "T-TPT"]
CATS = ["AUSENTE", "CONSTANTE", "CONGELADA", "QUANTIZADA", "CONTINUA"]
CAT_EN = ["Absent", "Constant", "Frozen", "Quantised", "Continuous"]
CAT_COL = ["#d9d9d9", "#c94040", "#e8a33d", "#4d7ea8", "#3f8f5f"]

VARIABLES = [
    "ABER-CKGL", "ABER-CKP", "ESTADO-DHSV", "ESTADO-M1", "ESTADO-M2",
    "ESTADO-PXO", "ESTADO-SDV-GL", "ESTADO-SDV-P", "ESTADO-W1", "ESTADO-W2",
    "ESTADO-XO", "P-ANULAR", "P-JUS-BS", "P-JUS-CKGL", "P-JUS-CKP",
    "P-MON-CKGL", "P-MON-CKP", "P-MON-SDV-P", "P-PDG", "PT-P", "P-TPT",
    "QBS", "QGL", "T-JUS-CKP", "T-MON-CKP", "T-PDG", "T-TPT",
]

CM = 1 / 2.54
W1, W2 = 8.5 * CM, 17.5 * CM     # coluna simples / dupla


def load(name):
    p = Path(name)
    if not p.exists():
        print(f"  [pulado] {name} nao encontrado")
        return None
    return pl.read_parquet(p)


# ---------------------------------------------------------------------------
def fig1_well_class(mf, out):
    real = mf.filter(pl.col("instance_type") == "real")
    wells = sorted({w for w in real["well_id"].to_list() if w})
    M = np.zeros((len(wells), 10))
    idx = {w: i for i, w in enumerate(wells)}
    for w, c in zip(real["well_id"].to_list(), real["event_class_dir"].to_list()):
        if w:
            M[idx[w], c] += 1

    fig, ax = plt.subplots(figsize=(W1, 0.16 * len(wells) + 1.0))
    Mm = np.ma.masked_where(M == 0, M)
    im = ax.imshow(Mm, aspect="auto", cmap="viridis",
                   norm=LogNorm(vmin=1, vmax=max(M.max(), 2)))
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticks(range(len(wells)))
    ax.set_yticklabels([w.replace("WELL-000", "W") for w in wells])
    ax.set_xlabel("Event class")
    ax.set_ylabel("Well")
    ax.set_facecolor("#f7f7f7")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Instances", fontsize=7)
    fig.savefig(out / "fig1_well_class.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig2_availability(mf, out):
    rates = {}
    for dom in ("real", "simulated"):
        s = mf.filter(pl.col("instance_type") == dom)
        rates[dom] = [float(s[f"{v}__present"].mean()) for v in VARIABLES]

    order = np.argsort(-(np.array(rates["real"]) + np.array(rates["simulated"])))
    names = [VARIABLES[i] for i in order]
    r = np.array(rates["real"])[order]
    s = np.array(rates["simulated"])[order]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(W1, 0.20 * len(names) + 0.8))
    ax.barh(y - 0.2, r, height=0.4, color="#2b6cb0", label="Real")
    ax.barh(y + 0.2, s, height=0.4, color="#c05621", label="Simulated")
    for i, (a, b) in enumerate(zip(r, s)):
        if a > 0.5 and b > 0.5:
            ax.text(1.02, i, "\u2605", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Fraction of instances in which the variable is present")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(out / "fig2_availability.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig3_missingness(mf, out):
    edges = [(-.001, .0001), (.0001, .01), (.01, .10),
             (.10, .50), (.50, .99), (.99, .9999), (.9999, 1.001)]
    labels = ["0", "(0,.01]", "(.01,.1]", "(.1,.5]", "(.5,.99]", "(.99,1)", "1"]

    fig, ax = plt.subplots(figsize=(W1, 4.5 * CM))
    x = np.arange(len(labels))
    for k, (dom, col, off) in enumerate([("real", "#2b6cb0", -0.2),
                                         ("simulated", "#c05621", 0.2)]):
        s = mf.filter(pl.col("instance_type") == dom)
        vals = np.concatenate([s[f"{v}__missing_frac"].fill_null(1.0)
                               .to_numpy().astype(float) for v in VARIABLES])
        h = [100 * np.mean((vals > lo) & (vals <= hi)) for lo, hi in edges]
        ax.bar(x + off, h, width=0.4, color=col,
               label="Real" if k == 0 else "Simulated")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Fraction of missing observations within an instance")
    ax.set_ylabel("% of (instance, variable) pairs")
    ax.legend(frameon=False)
    fig.savefig(out / "fig3_missingness.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig4_taxonomy(tx, out):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 5.5 * CM),
                             gridspec_kw={"width_ratios": [1, 2.2]})

    ax = axes[0]
    doms, bottom = ["real", "simulated"], np.zeros(2)
    for c, lab, col in zip(CATS, CAT_EN, CAT_COL):
        vals = []
        for d in doms:
            s = tx.filter(pl.col("instance_type") == d)
            vals.append(100 * s.filter(pl.col("category") == c).height / s.height)
        ax.bar(["Real", "Simulated"], vals, bottom=bottom, color=col, label=lab)
        bottom += np.array(vals)
    ax.set_ylabel("% of (instance, variable) pairs")
    ax.set_ylim(0, 100)

    ax = axes[1]
    real = tx.filter(pl.col("instance_type") == "real")
    vars_ = [v for v in SHARED + ["P-ANULAR", "P-JUS-CKGL", "QGL", "T-PDG"]
             if real.filter(pl.col("variable") == v).height]
    x = np.arange(len(vars_))
    bottom = np.zeros(len(vars_))
    for c, lab, col in zip(CATS, CAT_EN, CAT_COL):
        vals = []
        for v in vars_:
            s = real.filter(pl.col("variable") == v)
            vals.append(100 * s.filter(pl.col("category") == c).height / s.height)
        ax.bar(x, vals, bottom=bottom, color=col, label=lab, width=0.72)
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(vars_, rotation=35, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of real instances")
    ax.legend(ncol=5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.22))
    fig.savefig(out / "fig4_taxonomy.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig5_leakage(res, seeds, out):
    protos = ["random_window", "random_instance", "lowo"]
    plabels = ["Random\nwindow", "Random\ninstance", "Leave-one\nwell-out"]
    models = ["threshold", "logreg", "hgb"]
    mlabels = ["Threshold rule", "Logistic regression", "Gradient boosting"]
    cols = ["#8d8d8d", "#2b6cb0", "#c05621"]

    fig, ax = plt.subplots(figsize=(W1, 5.5 * CM))
    x = np.arange(len(protos))
    for k, (m, lab, col) in enumerate(zip(models, mlabels, cols)):
        vals = []
        for p in protos:
            s = res.filter((pl.col("protocol") == p) & (pl.col("model") == m)
                           & (pl.col("norm") == "perwell")
                           & pl.col("auroc").is_not_null())
            vals.append(s["auroc"].mean() if s.height else np.nan)
        ax.plot(x, vals, "o-", color=col, label=lab, lw=1.4, ms=4)
        for xi, vi in zip(x, vals):
            if np.isfinite(vi):
                ax.annotate(f"{vi:.3f}", (xi, vi), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=6, color=col)
    ax.axhline(0.5, ls=":", c="k", lw=0.8)
    ax.text(2.05, 0.505, "chance", fontsize=6, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(plabels)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel("Validation protocol")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(out / "fig5_leakage.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig6_folds(res, out):
    sub = res.filter((pl.col("protocol") == "lowo")
                     & (pl.col("norm") == "perwell")
                     & pl.col("auroc").is_not_null())
    folds = sorted(sub["fold"].unique().to_list())
    models = ["logreg", "hgb"]
    mlabels = ["Logistic regression", "Gradient boosting"]
    cols = ["#2b6cb0", "#c05621"]

    fig, ax = plt.subplots(figsize=(W1, 5.0 * CM))
    x = np.arange(len(folds))
    for k, (m, lab, col) in enumerate(zip(models, mlabels, cols)):
        vals = [sub.filter((pl.col("fold") == f) & (pl.col("model") == m))["auroc"].mean()
                for f in folds]
        ax.scatter(x + (k - 0.5) * 0.18, vals, s=26, color=col, label=lab, zorder=3)
        mu = np.nanmean(np.array(vals, dtype=float))
        ax.axhline(mu, color=col, ls="--", lw=0.8, alpha=0.6)
    prev = [sub.filter(pl.col("fold") == f)["prev"][0] for f in folds]
    ax.axhline(0.5, ls=":", c="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f.replace('WELL-000','W')}\n({p:.0%})"
                        for f, p in zip(folds, prev)])
    ax.set_xlabel("Held-out well (positive-class prevalence)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.2, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "fig6_folds.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig7_interpolation(itp, mf, root, out):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 5.0 * CM),
                            gridspec_kw={"width_ratios": [1, 1.5]})

    ax = axes[0]
    groups = [("real_c9", "Real\n(class 9)"), ("real_outros", "Real\n(other)"),
              ("simulado", "Simulated")]
    data = [itp.filter(pl.col("grp") == g)["tolerant_interp"].to_numpy()
            for g, _ in groups]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([l for _, l in groups])
    for patch, c in zip(bp["boxes"], ["#2b6cb0", "#5b8fc9", "#c05621"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("k")
    ax.set_ylabel("Fraction of samples with $|\\Delta^2 x|\\leq q$")
    ax.set_ylim(0, 1.02)

    # trecho ilustrativo: rampas lineares no sinal real
    ax = axes[1]
    c9 = mf.filter((pl.col("event_class_dir") == 9)
                   & (pl.col("instance_type") == "real"))
    drawn = False
    for r in c9.to_dicts():
        fp = Path(r["filepath"])
        if not fp.exists():
            fp = root / "9" / fp.name
        if not fp.exists():
            continue
        try:
            df = pl.read_parquet(fp)
        except Exception:
            continue
        if "P-MON-CKP" not in df.columns:
            continue
        x = df["P-MON-CKP"].to_numpy().astype(float)
        ok = np.isfinite(x)
        if ok.sum() < 5000:
            continue
        i0 = int(np.flatnonzero(ok)[2000])
        seg = x[i0:i0 + 300]
        if not np.all(np.isfinite(seg)):
            continue
        ax.plot(np.arange(seg.size), seg, "-", color="#2b6cb0", lw=0.8)
        ax.plot(np.arange(seg.size), seg, ".", color="#2b6cb0", ms=2)
        ax.set_xlabel("Time (s, nominal 1 Hz)")
        ax.set_ylabel("P-MON-CKP (Pa)")
        ax.set_title(f"Real instance, {r['well_id']}", fontsize=7)
        drawn = True
        break
    if not drawn:
        ax.text(0.5, 0.5, "no illustrative segment available",
                ha="center", va="center", transform=ax.transAxes)
    fig.savefig(out / "fig7_interpolation.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    root, out = Path("."), Path("figs")
    for a in sys.argv[1:]:
        if a.startswith("--raiz"):
            root = Path(a.split("=", 1)[1])
        elif a.startswith("--out"):
            out = Path(a.split("=", 1)[1])
    out.mkdir(parents=True, exist_ok=True)

    mf = load("manifest_3w.parquet")
    tx = load("taxonomy_pairs.parquet")
    res = load("results.parquet")
    seeds = load("results_seeds.parquet")
    itp = load("interpolation_test.parquet")

    jobs = [
        ("fig1", lambda: fig1_well_class(mf, out), mf is not None),
        ("fig2", lambda: fig2_availability(mf, out), mf is not None),
        ("fig3", lambda: fig3_missingness(mf, out), mf is not None),
        ("fig4", lambda: fig4_taxonomy(tx, out), tx is not None),
        ("fig5", lambda: fig5_leakage(res, seeds, out), res is not None),
        ("fig6", lambda: fig6_folds(res, out), res is not None),
        ("fig7", lambda: fig7_interpolation(itp, mf, root, out),
         itp is not None and mf is not None),
    ]
    for name, fn, ok in jobs:
        if not ok:
            print(f"{name}: entrada ausente, pulado")
            continue
        try:
            fn()
            print(f"{name}: ok")
        except Exception as e:
            print(f"{name}: FALHOU -> {e!r}")

    print(f"\nFiguras em {out.resolve()}")


if __name__ == "__main__":
    main()