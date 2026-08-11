#!/usr/bin/env python3
"""
step8_experiment.py  (versao corrigida)
=======================================
Experimento central: mede (i) a diferenca entre protocolos de validacao e
(ii) o efeito do esquema de normalizacao.

CORRECOES EM RELACAO A VERSAO ANTERIOR
  (1) 'random_instance' usava GroupKFold direto sobre as instancias, o que e
      DETERMINISTICO: nao era um sorteio, era uma particao fixa e arbitraria.
      Aquela particao especifica favorecia o modelo mais flexivel.
      -> agora a atribuicao de instancias a dobras depende da semente.
  (2) Uma unica semente. Um sorteio so nao caracteriza um protocolo aleatorio.
      -> repete os protocolos aleatorios sobre N sementes e registra a semente.
  (3) As features eram recalculadas a cada dobra mesmo para normalizacoes que
      nao dependem do treino.
      -> 'none' e 'perwell' sao calculadas uma vez; so 'global' e por dobra.

PROTOCOLOS
  random_window   - 5-fold estratificado sobre janelas. Vaza: janelas
                    sobrepostas do mesmo evento caem em treino e teste.
  random_instance - 5-fold agrupado por instancia. Corrige a sobreposicao,
                    nao o vazamento por poco.
  lowo            - leave-one-well-out. Honesto. Particao fixa por construcao.

NORMALIZACAO
  none     - series brutas; o nivel absoluto identifica o poco
  global   - z-score com estatisticas dos POCOS DE TREINO
  perwell  - z-score com estatisticas do proprio poco (sem usar rotulos)

MODELOS
  majority   - classe majoritaria do treino
  threshold  - limiar sobre a ultima leitura de T-JUS-CKP, ajustado no treino
  logreg     - regressao logistica sobre estatisticas da janela
  hgb        - histogram gradient boosting sobre as mesmas estatisticas

Requer: pip install scikit-learn

Uso:
    python step8_experiment.py [windows_class9.npz] [--seeds=10] [--out-dir=.]
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (average_precision_score, f1_score,
                                 roc_auc_score)
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    print("ERRO: pip install scikit-learn")
    sys.exit(1)

LINE = "=" * 78
MIN_TEST = 10          # minimo de exemplos de cada classe no teste
N_FOLDS = 5

PROTOCOLS = ["random_window", "random_instance", "lowo"]
SCHEMES = ["none", "global", "perwell"]
MODELS = ["majority", "threshold", "logreg", "hgb"]


def sec(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


# ---------------------------------------------------------------------------
# Features e normalizacao
# ---------------------------------------------------------------------------

def window_features(X):
    """X (n, T, V) -> (n, V*7) estatisticas por janela."""
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
    """tr = indices de treino (usado apenas por 'global')."""
    Xn = X.astype(np.float64)
    if scheme == "none":
        return Xn.copy()
    if scheme == "global":
        mu = Xn[tr].mean(axis=(0, 1), keepdims=True)
        sd = Xn[tr].std(axis=(0, 1), keepdims=True) + 1e-9
        return (Xn - mu) / sd
    if scheme == "perwell":
        out = Xn.copy()
        for w in np.unique(well):
            m = well == w
            mu = out[m].mean(axis=(0, 1), keepdims=True)
            sd = out[m].std(axis=(0, 1), keepdims=True) + 1e-9
            out[m] = (out[m] - mu) / sd
        return out
    raise ValueError(scheme)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def best_threshold(score, y):
    """Limiar e sentido que maximizam F1 no TREINO."""
    cand = np.quantile(score, np.linspace(0.01, 0.99, 99))
    best, bsign, bf = cand[0], 1, -1.0
    for c in cand:
        for sign in (1, -1):
            f = f1_score(y, (sign * score >= sign * c).astype(int),
                         zero_division=0)
            if f > bf:
                bf, best, bsign = f, c, sign
    return best, bsign


def run_model(name, Ftr, ytr, Fte, Xtr_raw, Xte_raw, seed):
    """Score continuo no teste (maior = mais provavel positivo)."""
    if name == "majority":
        return np.full(len(Fte), float(ytr.mean()))

    if name == "threshold":
        # ultima leitura de T-JUS-CKP (indice 1 no eixo de variaveis)
        thr, sign = best_threshold(Xtr_raw[:, -1, 1], ytr)
        return sign * (Xte_raw[:, -1, 1] - thr)

    if len(np.unique(ytr)) < 2:
        return np.full(len(Fte), float(ytr.mean()))

    if name == "logreg":
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif name == "hgb":
        m = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.1,
            random_state=seed, early_stopping=False)
    else:
        raise ValueError(name)
    m.fit(Ftr, ytr)
    return m.predict_proba(Fte)[:, 1]


def evaluate(y, score):
    out = {"n": len(y), "prev": float(np.mean(y))}
    if len(np.unique(y)) < 2:
        return {**out, "auroc": None, "auprc": None, "f1": None}
    out["auroc"] = float(roc_auc_score(y, score))
    out["auprc"] = float(average_precision_score(y, score))
    qs = np.quantile(score, np.linspace(0.01, 0.99, 99))
    out["f1"] = float(max(f1_score(y, (score >= q).astype(int),
                                   zero_division=0) for q in qs))
    return out


# ---------------------------------------------------------------------------
# Protocolos
# ---------------------------------------------------------------------------

def make_folds(protocol, y, well, inst, seed):
    idx = np.arange(len(y))

    if protocol == "random_window":
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        return [(tr, te, f"f{i}") for i, (tr, te) in enumerate(cv.split(idx, y))]

    if protocol == "random_instance":
        # CORRECAO: atribuicao de instancias a dobras dependente da semente
        rng = np.random.default_rng(seed)
        insts = np.unique(inst)
        rng.shuffle(insts)
        fold_of = {name: i % N_FOLDS for i, name in enumerate(insts)}
        assign = np.array([fold_of[x] for x in inst])
        return [(idx[assign != k], idx[assign == k], f"f{k}")
                for k in range(N_FOLDS)]

    if protocol == "lowo":
        folds = []
        for w in sorted(np.unique(well)):
            te = idx[well == w]
            if (y[te] == 1).sum() < MIN_TEST or (y[te] == 0).sum() < MIN_TEST:
                continue
            folds.append((idx[well != w], te, str(w)))
        return folds

    raise ValueError(protocol)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pos, out_dir, n_seeds = [], Path("."), 10
    for a in sys.argv[1:]:
        if a.startswith("--out-dir"):
            out_dir = Path(a.split("=", 1)[1])
        elif a.startswith("--seeds"):
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
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(npz, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    well, inst = d["well"], d["instance"]
    variables = [str(v) for v in d["variables"]]

    print(f"janelas: {X.shape}   positivas: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"variaveis: {variables}")
    print(f"pocos: {len(np.unique(well))}   instancias: {len(np.unique(inst))}")
    print(f"sementes: {n_seeds}\n")

    # (3) features que nao dependem do treino: calculadas uma unica vez
    all_idx = np.arange(len(y))
    F_cache = {s: window_features(normalize(X, well, all_idx, s))
               for s in ("none", "perwell")}

    rows = []
    for seed in range(n_seeds):
        for proto in PROTOCOLS:
            folds = make_folds(proto, y, well, inst, seed)
            for scheme in SCHEMES:
                for tr, te, tag in folds:
                    if scheme == "global":
                        F = window_features(normalize(X, well, tr, "global"))
                    else:
                        F = F_cache[scheme]
                    for mdl in MODELS:
                        score = run_model(mdl, F[tr], y[tr], F[te],
                                          X[tr], X[te], seed)
                        m = evaluate(y[te], score)
                        rows.append({"seed": seed, "protocol": proto,
                                     "norm": scheme, "model": mdl,
                                     "fold": str(tag), **m})
        print(f"  semente {seed + 1}/{n_seeds}")

    res = pl.DataFrame(rows, infer_schema_length=None)
    res.write_parquet(out_dir / "results.parquet")
    res.write_csv(out_dir / "results.csv")

    def agg(proto, scheme, mdl):
        s = res.filter((pl.col("protocol") == proto) & (pl.col("norm") == scheme)
                       & (pl.col("model") == mdl) & pl.col("auroc").is_not_null())
        if not s.height:
            return None, None
        per_seed = (s.group_by("seed").agg(pl.col("auroc").mean())
                     .sort("seed")["auroc"].to_numpy())
        return float(per_seed.mean()), float(per_seed.std())

    sec("AUROC MEDIO (media sobre sementes) POR PROTOCOLO x NORM x MODELO")
    hdr = f"{'protocolo':<18}{'norm':<10}" + "".join(f"{m:>14}" for m in MODELS)
    print(hdr)
    print("-" * len(hdr))
    for proto in PROTOCOLS:
        for scheme in SCHEMES:
            cells = ""
            for mdl in MODELS:
                mu, sd = agg(proto, scheme, mdl)
                cells += f"{mu:>9.3f}±{sd:<4.3f}" if mu is not None else f"{'-':>14}"
            print(f"{proto:<18}{scheme:<10}{cells}")
        print()

    sec("ORCAMENTO DE VAZAMENTO (norm=perwell)")
    print(f"{'modelo':<12}{'rand.window':>14}{'rand.instance':>16}"
          f"{'lowo':>10}{'sobreposicao':>14}{'poco':>8}{'total':>8}")
    print("-" * 82)
    for mdl in MODELS:
        a, _ = agg("random_window", "perwell", mdl)
        b, _ = agg("random_instance", "perwell", mdl)
        c, _ = agg("lowo", "perwell", mdl)
        if None in (a, b, c):
            continue
        print(f"{mdl:<12}{a:>14.3f}{b:>16.3f}{c:>10.3f}"
              f"{a-b:>14.3f}{b-c:>8.3f}{a-c:>8.3f}")
    print("\n'sobreposicao' = janelas sobrepostas; 'poco' = identidade do poco.")

    sec("LOWO POR DOBRA (norm=perwell)")
    sub = res.filter((pl.col("protocol") == "lowo") & (pl.col("norm") == "perwell"))
    print(f"{'dobra':<16}{'n':>7}{'prev':>7}" + "".join(f"{m:>13}" for m in MODELS))
    print("-" * (30 + 13 * len(MODELS)))
    for fold in sorted(sub["fold"].unique().to_list()):
        f0 = sub.filter(pl.col("fold") == fold)
        cells = ""
        for mdl in MODELS:
            s = f0.filter(pl.col("model") == mdl)
            v = s["auroc"].mean()
            cells += f"{v:>13.3f}" if v is not None else f"{'-':>13}"
        print(f"{fold:<16}{int(f0['n'][0]):>7}{float(f0['prev'][0]):>7.2f}{cells}")

    print(f"\nSalvo: {out_dir/'results.parquet'} e results.csv")
    print("\nLembretes:")
    print("  - AUPRC depende da prevalencia; reporte sempre com prev ao lado")
    print("  - o F1 usa o melhor limiar do TESTE: e limite superior otimista")
    print("  - 2582 janelas sobrepostas correspondem a ~7 eventos distintos")
    print("  - LOWO tem particao fixa; o desvio entre sementes reflete apenas")
    print("    a semente do modelo, e deve ser proximo de zero")
    print()


if __name__ == "__main__":
    main()