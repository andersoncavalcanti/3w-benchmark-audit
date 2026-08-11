#!/usr/bin/env python3
"""
step7_build_windows.py
======================
Constroi o conjunto de janelas para deteccao do transiente de hidrato
(rotulo 109) contra operacao normal (rotulo 0), na classe 9 real.

PROTOCOLO
  variaveis .......... P-MON-CKP, T-JUS-CKP  (par termodinamico do choke)
  reamostragem ....... 1 Hz -> 1/60 Hz, media por minuto; rotulo = ULTIMA
                       observacao do minuto (causal)
  janela ............. 60 min, passo 5 min
  rotulo da janela ... rotulo da observacao FINAL (deteccao causal)
  exclusoes .......... janelas com qualquer rotulo 9 (estado permanente,
                       apenas 4,9 h no total), janelas com lacuna temporal,
                       janelas com qualquer NaN
  normalizacao ....... NAO aplicada aqui. Deve ser calculada por dobra,
                       somente sobre o treino (z-score por poco).

SAIDAS
  windows_class9.npz     X (n,60,2), y, well, instance, t_end
  windows_index.parquet  metadados por janela (sem X)

Uso:
    python step7_build_windows.py <raiz> [viabilidade] [manifesto] [opcoes]

Opcoes:
    --vars=P-MON-CKP,T-JUS-CKP   subconjunto de variaveis
    --win=60                     comprimento da janela em minutos
    --stride=5                   passo em minutos
    --out-dir=CAMINHO
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_VARS = ["P-MON-CKP", "T-JUS-CKP"]
USEFUL = {"QUANTIZADA", "CONTINUA"}

ALL_VARIABLES = [
    "ABER-CKGL", "ABER-CKP",
    "ESTADO-DHSV", "ESTADO-M1", "ESTADO-M2", "ESTADO-PXO",
    "ESTADO-SDV-GL", "ESTADO-SDV-P", "ESTADO-W1", "ESTADO-W2", "ESTADO-XO",
    "P-ANULAR", "P-JUS-BS", "P-JUS-CKGL", "P-JUS-CKP",
    "P-MON-CKGL", "P-MON-CKP", "P-MON-SDV-P", "P-PDG",
    "PT-P", "P-TPT",
    "QBS", "QGL",
    "T-JUS-CKP", "T-MON-CKP", "T-PDG", "T-TPT",
]

POS_LABEL, NEG_LABEL, EXCLUDE_LABEL = 109, 0, 9
LINE = "=" * 78


def sec(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


def find_cols(df):
    ts = [c for c, d in zip(df.columns, df.dtypes) if d in (pl.Datetime, pl.Date)]
    labels = [c for c in df.columns
              if c not in ALL_VARIABLES and c not in ts
              and not c.startswith("__index")]
    cls = None
    for c in labels:
        lc = c.lower()
        if "state" in lc or "estado" in lc:
            continue
        if "class" in lc or "classe" in lc or lc == "label":
            cls = c
    if cls is None and labels:
        cls = labels[0]
    return (ts[0] if ts else None), cls


def resample_minute(df, tcol, cls_col, variables):
    """1 Hz -> 1 min. Media das variaveis, ultimo rotulo do minuto."""
    aggs = [pl.col(v).mean().alias(v) for v in variables]
    aggs.append(pl.col(cls_col).last().alias("label"))
    aggs.append(pl.len().alias("n_raw"))
    return (df.sort(tcol)
              .group_by_dynamic(tcol, every="1m")
              .agg(aggs)
              .sort(tcol))


def build_windows(m, tcol, variables, win, stride):
    """Janelas contiguas, sem NaN, sem rotulo 9. Rotulo = ultima obs."""
    t = m[tcol].to_numpy()
    minute_idx = ((t - t[0]).astype("timedelta64[s]").astype(np.int64)) // 60
    V = np.column_stack([m[v].to_numpy().astype(float) for v in variables])
    lab = m["label"].to_numpy()

    X, y, tend, drops = [], [], [], {"gap": 0, "nan": 0, "label9": 0,
                                     "other_label": 0}
    n = len(m)
    for s in range(0, n - win + 1, stride):
        e = s + win
        if minute_idx[e - 1] - minute_idx[s] != win - 1:
            drops["gap"] += 1
            continue
        seg_lab = lab[s:e]
        if np.any(seg_lab == EXCLUDE_LABEL):
            drops["label9"] += 1
            continue
        final = seg_lab[-1]
        if final not in (POS_LABEL, NEG_LABEL):
            drops["other_label"] += 1
            continue
        seg = V[s:e]
        if not np.all(np.isfinite(seg)):
            drops["nan"] += 1
            continue
        X.append(seg)
        y.append(1 if final == POS_LABEL else 0)
        tend.append(t[e - 1])
    return X, y, tend, drops


def main():
    pos_args, opts = [], {"win": 60, "stride": 5,
                          "vars": ",".join(DEFAULT_VARS), "out-dir": "."}
    for a in sys.argv[1:]:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            if k not in opts:
                print(f"Opcao desconhecida: {a}")
                sys.exit(1)
            opts[k] = int(v) if k in ("win", "stride") else v
        else:
            pos_args.append(a)

    if not pos_args:
        print(__doc__)
        sys.exit(1)

    root = Path(pos_args[0]).expanduser().resolve()
    via = Path(pos_args[1]) if len(pos_args) > 1 else Path("class9_viability.parquet")
    man = Path(pos_args[2]) if len(pos_args) > 2 else Path("manifest_3w.parquet")
    out_dir = Path(opts["out-dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (via, man):
        if not p.exists():
            print(f"ERRO: nao encontrado: {p}")
            sys.exit(1)

    variables = [v.strip() for v in opts["vars"].split(",") if v.strip()]
    win, stride = opts["win"], opts["stride"]

    df = pl.read_parquet(via)
    mf = pl.read_parquet(man)
    paths = {Path(r["filepath"]).name: r["filepath"]
             for r in mf.select(["filepath"]).to_dicts()}

    elig = [r for r in df.to_dicts()
            if all(r.get(f"cat_{v}") in USEFUL for v in variables)]

    print(f"variaveis .: {', '.join(variables)}")
    print(f"janela ....: {win} min, passo {stride} min")
    print(f"instancias elegiveis: {len(elig)} em "
          f"{len({r['well_id'] for r in elig})} pocos\n")

    Xs, ys, wells, insts, tends = [], [], [], [], []
    all_drops = {"gap": 0, "nan": 0, "label9": 0, "other_label": 0}
    per_inst = []

    for r in sorted(elig, key=lambda x: (x["well_id"], x["filename"])):
        fn = r["filename"]
        fp = Path(paths.get(fn, ""))
        if not fp.exists():
            fp = root / "9" / fn
        if not fp.exists():
            print(f"  AVISO: nao encontrado: {fn}")
            continue

        raw = pl.read_parquet(fp)
        tcol, cls_col = find_cols(raw)
        if tcol is None or cls_col is None:
            print(f"  AVISO: colunas nao identificadas em {fn}")
            continue
        miss = [v for v in variables if v not in raw.columns]
        if miss:
            print(f"  AVISO: {fn} sem {miss}")
            continue

        m = resample_minute(raw.select([tcol, cls_col] + variables),
                            tcol, cls_col, variables)
        X, y, tend, drops = build_windows(m, tcol, variables, win, stride)
        for k in all_drops:
            all_drops[k] += drops[k]

        if X:
            Xs.extend(X)
            ys.extend(y)
            wells.extend([r["well_id"]] * len(X))
            insts.extend([fn] * len(X))
            tends.extend(tend)

        per_inst.append({
            "well_id": r["well_id"], "filename": fn,
            "minutes": m.height, "windows": len(X),
            "pos": int(sum(y)), "neg": int(len(y) - sum(y)),
        })
        print(f"  {r['well_id']}  {fn[:38]:<39} "
              f"{m.height:>6} min -> {len(X):>5} jan "
              f"({sum(y):>4} pos / {len(y)-sum(y):>4} neg)")

    if not Xs:
        print("\nERRO: nenhuma janela gerada.")
        sys.exit(1)

    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int8)
    well = np.array(wells)
    inst = np.array(insts)
    tend = np.array(tends)

    np.savez_compressed(out_dir / "windows_class9.npz",
                        X=X, y=y, well=well, instance=inst,
                        t_end=tend.astype("datetime64[s]").astype(np.int64),
                        variables=np.array(variables),
                        win=win, stride=stride)
    pl.DataFrame(per_inst).write_parquet(out_dir / "windows_index.parquet")

    sec("RESUMO")
    print(f"  janelas .............. {X.shape[0]}")
    print(f"  formato de X ......... {X.shape}  (janelas, minutos, variaveis)")
    print(f"  positivas (109) ...... {int(y.sum())} "
          f"({100*y.mean():.1f}%)")
    print(f"  negativas (0) ........ {int((y == 0).sum())}")
    print(f"  pocos ................ {len(np.unique(well))}")
    print(f"\n  descartadas: lacuna={all_drops['gap']}  "
          f"NaN={all_drops['nan']}  rotulo9={all_drops['label9']}  "
          f"outro_rotulo={all_drops['other_label']}")

    sec("DOBRAS LEAVE-ONE-WELL-OUT")
    print(f"{'poco (dobra teste)':<20}{'janelas':>9}{'pos':>7}{'neg':>7}"
          f"{'%pos':>8}   viavel como teste?")
    print("-" * 74)
    usable = 0
    for w in sorted(np.unique(well)):
        mk = well == w
        p, ng = int(y[mk].sum()), int((y[mk] == 0).sum())
        ok = p >= 10 and ng >= 10
        usable += ok
        print(f"{w:<20}{mk.sum():>9}{p:>7}{ng:>7}"
              f"{100*p/mk.sum():>7.1f}%   {'sim' if ok else 'so treino'}")
    print(f"\n  dobras de teste utilizaveis: {usable}")
    if usable < 5:
        print("  ATENCAO: menos de 5 dobras com ambas as classes.")
        print("  Considere reduzir o passo para gerar mais janelas, ou")
        print("  reportar as dobras individualmente sem media agregada.")

    sec("PROXIMOS PASSOS")
    print("  1. Normalizacao z-score por poco, calculada SOMENTE no treino")
    print("     de cada dobra. Nunca sobre o conjunto completo.")
    print("  2. Dois protocolos sobre os MESMOS dados:")
    print("       (a) split aleatorio de janelas  -> otimista, com vazamento")
    print("       (b) leave-one-well-out          -> honesto")
    print("     A diferenca entre (a) e (b) e o resultado central do artigo.")
    print("  3. Baselines antes de qualquer rede: regra de limiar sobre")
    print("     T-JUS-CKP, regressao logistica sobre estatisticas da janela,")
    print("     gradient boosting. So depois modelos sequenciais.")
    print(f"\n  Salvo em: {out_dir.resolve()}")
    print()


if __name__ == "__main__":
    main()