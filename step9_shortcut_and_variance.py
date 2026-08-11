#!/usr/bin/env python3
"""
step9_shortcut_and_variance.py
==============================
Tres analises que faltam antes de escrever:

  (A) IDENTIFICABILIDADE DO POCO  [teste direto do atalho]
      A ablacao de normalizacao do passo 8 nao isola o atalho: normalizar
      remove o atalho E melhora a transferencia, efeitos opostos.
      Aqui a medida e direta: treinar um classificador para prever o POCO
      a partir das mesmas features. Agrupado por instancia, para que o
      acerto nao venha de janelas sobrepostas.
      Se a acuracia superar muito o acaso, o poco e identificavel — e
      qualquer protocolo que nao isole poco esta exposto ao atalho.

  (B) VARIANCIA ENTRE SEMENTES
      Repete random_window e random_instance com varias sementes; LOWO
      tem particao fixa, mas o HGB depende da semente.

  (C) COMPARACAO PAREADA POR DOBRA
      logreg vs hgb sob LOWO, dobra a dobra. Com 6 dobras (~7 eventos),
      nenhum teste tem poder: o objetivo e mostrar a dispersao, nao
      alegar significancia.

Uso:
    python step9_shortcut_and_variance.py [windows_class9.npz] [--seeds=10]
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

LINE = "=" * 78
MIN_TEST = 10


def sec(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


def window_features(X):
    n, T, V = X.shape
    t = np.arange(T, dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    feats = []
    for v in range(V):
        s = X[:, :, v]
        feats += [s.mean(1), s.std(1), s.min(1), s.max(1),
                  s[:, -1], s[:, -1] - s[:, 0], (s * t).mean(1)]
    return np.column_stack(feats)


def normalize(X, well, tr, scheme):
    Xn = X.astype(np.float64).copy()
    if scheme == "none":
        return Xn
    if scheme == "global":
        mu = Xn[tr].mean(axis=(0, 1), keepdims=True)
        sd = Xn[tr].std(axis=(0, 1), keepdims=True) + 1e-9
        return (Xn - mu) / sd
    if scheme == "perwell":
        for w in np.unique(well):
            m = well == w
            mu = Xn[m].mean(axis=(0, 1), keepdims=True)
            sd = Xn[m].std(axis=(0, 1), keepdims=True) + 1e-9
            Xn[m] = (Xn[m] - mu) / sd
        return Xn
    raise ValueError(scheme)


# ---------------------------------------------------------------------------
# (A) Identificabilidade do poco
# ---------------------------------------------------------------------------

def well_identifiability(X, well, inst, seed):
    sec("(A) O POCO E IDENTIFICAVEL A PARTIR DO SINAL?")
    print("Classificacao multiclasse do poco, validada por GroupKFold")
    print("agrupando por INSTANCIA (janelas sobrepostas nunca cruzam folds).\n")

    wells, ycode = np.unique(well, return_inverse=True)
    chance_uniform = 1.0 / len(wells)
    counts = np.bincount(ycode)
    chance_majority = counts.max() / counts.sum()

    print(f"  poços .............. {len(wells)}")
    print(f"  acaso uniforme ..... {chance_uniform:.3f}")
    print(f"  classe majoritaria . {chance_majority:.3f}\n")

    print(f"{'normalizacao':<16}{'acuracia':>12}{'vs majoritaria':>18}")
    print("-" * 46)
    results = {}
    for scheme in ("none", "global", "perwell"):
        idx = np.arange(len(well))
        Xn = normalize(X, well, idx, scheme)
        F = window_features(Xn)
        preds = np.zeros(len(idx), dtype=int)
        cv = GroupKFold(n_splits=5)
        for tr, te in cv.split(F, ycode, groups=inst):
            if len(np.unique(ycode[tr])) < 2:
                continue
            m = HistGradientBoostingClassifier(
                max_iter=200, max_depth=4, random_state=seed,
                early_stopping=False)
            m.fit(F[tr], ycode[tr])
            preds[te] = m.predict(F[te])
        acc = accuracy_score(ycode, preds)
        results[scheme] = acc
        print(f"{scheme:<16}{acc:>12.3f}{acc - chance_majority:>+18.3f}")

    print("\n  Leitura: acuracia muito acima da classe majoritaria significa")
    print("  que a identidade do poco esta codificada no sinal. Qualquer")
    print("  protocolo que nao isole poços deixa esse atalho disponivel.")
    if results["perwell"] > chance_majority + 0.2:
        print("\n  >>> Normalizacao por poco NAO elimina a identificabilidade.")
        print("      O atalho sobrevive a normalizacao.")
    return results


# ---------------------------------------------------------------------------
# (B) Variancia entre sementes
# ---------------------------------------------------------------------------

def make_folds(protocol, y, well, inst, seed):
    idx = np.arange(len(y))
    if protocol == "random_window":
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        return [(tr, te, f"f{i}") for i, (tr, te) in enumerate(cv.split(idx, y))]
    if protocol == "random_instance":
        rng = np.random.default_rng(seed)
        insts = np.unique(inst)
        rng.shuffle(insts)
        fold_of = {name: i % 5 for i, name in enumerate(insts)}
        assign = np.array([fold_of[x] for x in inst])
        return [(idx[assign != k], idx[assign == k], f"f{k}") for k in range(5)]
    if protocol == "lowo":
        folds = []
        for w in sorted(np.unique(well)):
            te = idx[well == w]
            if (y[te] == 1).sum() < MIN_TEST or (y[te] == 0).sum() < MIN_TEST:
                continue
            folds.append((idx[well != w], te, str(w)))
        return folds
    raise ValueError(protocol)


def fit_score(mdl, Ftr, ytr, Fte, seed):
    if len(np.unique(ytr)) < 2:
        return np.full(len(Fte), float(ytr.mean()))
    if mdl == "logreg":
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        m = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.1,
            random_state=seed, early_stopping=False)
    m.fit(Ftr, ytr)
    return m.predict_proba(Fte)[:, 1]


def seed_variance(X, y, well, inst, n_seeds):
    sec(f"(B) VARIANCIA ENTRE {n_seeds} SEMENTES  (norm=perwell)")
    rows = []
    for seed in range(n_seeds):
        Xn = normalize(X, well, np.arange(len(y)), "perwell")
        F = window_features(Xn)
        for proto in ("random_window", "random_instance", "lowo"):
            for tr, te, tag in make_folds(proto, y, well, inst, seed):
                if len(np.unique(y[te])) < 2:
                    continue
                for mdl in ("logreg", "hgb"):
                    s = fit_score(mdl, F[tr], y[tr], F[te], seed)
                    rows.append({"seed": seed, "protocol": proto,
                                 "model": mdl, "fold": tag,
                                 "auroc": float(roc_auc_score(y[te], s))})
        print(f"  semente {seed + 1}/{n_seeds}")

    res = pl.DataFrame(rows)
    print(f"\n{'protocolo':<18}{'modelo':<10}{'media':>9}{'dp':>8}"
          f"{'min':>8}{'max':>8}")
    print("-" * 61)
    for proto in ("random_window", "random_instance", "lowo"):
        for mdl in ("logreg", "hgb"):
            s = res.filter((pl.col("protocol") == proto)
                           & (pl.col("model") == mdl))
            if not s.height:
                continue
            # media por semente, depois dispersao entre sementes
            per_seed = (s.group_by("seed").agg(pl.col("auroc").mean())
                         .sort("seed")["auroc"].to_numpy())
            print(f"{proto:<18}{mdl:<10}{per_seed.mean():>9.3f}"
                  f"{per_seed.std():>8.3f}{per_seed.min():>8.3f}"
                  f"{per_seed.max():>8.3f}")
        print()
    print("dp = desvio padrao das MEDIAS por semente (nao das dobras)")
    return res


# ---------------------------------------------------------------------------
# (C) Comparacao pareada por dobra
# ---------------------------------------------------------------------------

def paired(res):
    sec("(C) logreg vs hgb SOB LOWO, DOBRA A DOBRA")
    lo = res.filter(pl.col("protocol") == "lowo")
    folds = sorted(lo["fold"].unique().to_list())
    print(f"{'dobra':<16}{'logreg':>12}{'hgb':>12}{'diferenca':>12}")
    print("-" * 52)
    diffs = []
    for f in folds:
        a = lo.filter((pl.col("fold") == f) & (pl.col("model") == "logreg"))["auroc"].mean()
        b = lo.filter((pl.col("fold") == f) & (pl.col("model") == "hgb"))["auroc"].mean()
        if a is None or b is None:
            continue
        diffs.append(a - b)
        print(f"{f:<16}{a:>12.3f}{b:>12.3f}{a - b:>+12.3f}")
    d = np.array(diffs)
    print("-" * 52)
    print(f"{'media':<16}{'':>12}{'':>12}{d.mean():>+12.3f}")
    print(f"{'dobras a favor':<16}{'':>12}{'':>12}"
          f"{int((d > 0).sum())}/{len(d):>11}")
    print(f"\nCom {len(d)} dobras (~7 eventos distintos), nenhum teste de")
    print("hipotese tem poder util. Reporte a dispersao, nao um valor-p.")


def main():
    pos, n_seeds = [], 10
    for a in sys.argv[1:]:
        if a.startswith("--seeds"):
            n_seeds = int(a.split("=", 1)[1])
        elif a.startswith("--"):
            print(f"Opcao desconhecida: {a}")
            sys.exit(1)
        else:
            pos.append(a)

    npz = Path(pos[0]) if pos else Path("windows_class9.npz")
    if not npz.exists():
        print(f"ERRO: nao encontrado: {npz}")
        sys.exit(1)

    d = np.load(npz, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    well, inst = d["well"], d["instance"]
    print(f"janelas: {X.shape}  positivas: {y.sum()} ({100*y.mean():.1f}%)")

    well_identifiability(X, well, inst, 42)
    res = seed_variance(X, y, well, inst, n_seeds)
    res.write_parquet("results_seeds.parquet")
    paired(res)
    print("\nSalvo: results_seeds.parquet\n")


if __name__ == "__main__":
    main()