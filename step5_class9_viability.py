#!/usr/bin/env python3
"""
step5_class9_viability.py
=========================
Duas verificacoes que decidem se a demonstracao empirica e viavel:

  (A) CONJUNTO UTILIZAVEL DA CLASSE 9
      As 29 instancias "completas" do passo 3 foram selecionadas apenas por
      ausencia de nulos. A taxonomia mostrou que ausencia de nulo nao implica
      informacao (24,9% dos pares nao-ausentes sao CONSTANTES).
      Aqui reavaliamos exigindo que a variavel seja QUANTIZADA ou CONTINUA.

  (B) INSTANCIAS DUPLICADAS ENTRE POCOS
      Contagens de observacoes identicas aparecem em pocos diferentes.
      Se forem duplicatas de fato, ha vazamento mesmo sob leave-one-well-out.
      Verificacao por hash do conteudo numerico.

Uso:
    python step5_class9_viability.py <raiz> [manifesto] [taxonomia] [--out-dir=X]
"""

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

SHARED = ["P-MON-CKP", "P-PDG", "P-TPT", "T-JUS-CKP", "T-TPT"]
USEFUL = {"QUANTIZADA", "CONTINUA"}
LINE = "=" * 78


def sec(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


# ---------------------------------------------------------------------------
# (A) Conjunto utilizavel
# ---------------------------------------------------------------------------

def viability(mf: pl.DataFrame, tx: pl.DataFrame):
    sec("(A) CLASSE 9 REAL: DE 'PRESENTE' PARA 'UTILIZAVEL'")

    c9 = mf.filter((pl.col("event_class_dir") == 9)
                   & (pl.col("instance_type") == "real"))
    txc9 = tx.filter((pl.col("event_class_dir") == 9)
                     & (pl.col("instance_type") == "real"))

    # categoria por (arquivo, variavel)
    cat = {}
    for r in txc9.select(["filename", "variable", "category"]).to_dicts():
        cat[(r["filename"], r["variable"])] = r["category"]

    print(f"{'poco':<13}{'arquivo':<34}{'h':>6}{'%c109':>7}"
          + "".join(f"{v[:9]:>11}" for v in SHARED) + "  uteis")
    print("-" * 128)

    recs = []
    for r in c9.sort(["well_id", "n_obs"]).to_dicts():
        fn = Path(r["filepath"]).name
        cats = [cat.get((fn, v), "AUSENTE") for v in SHARED]
        n_useful = sum(c in USEFUL for c in cats)
        f109 = (r.get("class_109_frac") or 0) * 100
        recs.append({
            "well_id": r["well_id"], "filename": fn,
            "n_obs": r["n_obs"], "f109": f109,
            "n_useful": n_useful,
            "positive": f109 > 0,
            **{f"cat_{v}": c for v, c in zip(SHARED, cats)},
        })
        print(f"{str(r['well_id']):<13}{fn[:33]:<34}"
              f"{(r['duration_s'] or 0)/3600:>6.1f}{f109:>7.1f}"
              + "".join(f"{c[:9]:>11}" for c in cats)
              + f"{n_useful:>7}")

    df = pl.DataFrame(recs)

    sec("(A2) QUANTAS INSTANCIAS POR NIVEL DE EXIGENCIA")
    print(f"{'exige >= k uteis':<20}{'instancias':>12}{'pocos':>8}"
          f"{'positivas':>11}{'negativas':>11}{'pocos c/ pos':>14}")
    print("-" * 76)
    for k in range(5, 0, -1):
        s = df.filter(pl.col("n_useful") >= k)
        pos = s.filter(pl.col("positive"))
        neg = s.filter(~pl.col("positive"))
        print(f"{'>= ' + str(k):<20}{s.height:>12}"
              f"{s['well_id'].n_unique():>8}{pos.height:>11}{neg.height:>11}"
              f"{pos['well_id'].n_unique():>14}")

    sec("(A3) UTILIDADE POR VARIAVEL DENTRO DA CLASSE 9")
    print(f"{'variavel':<14}{'util':>8}{'constante':>12}{'congelada':>12}"
          f"{'ausente':>10}")
    print("-" * 56)
    for v in SHARED:
        col = df[f"cat_{v}"].to_list()
        n = len(col)
        u = sum(c in USEFUL for c in col)
        k = col.count("CONSTANTE")
        f = col.count("CONGELADA")
        a = col.count("AUSENTE")
        print(f"{v:<14}{100*u/n:>7.1f}%{100*k/n:>11.1f}%"
              f"{100*f/n:>11.1f}%{100*a/n:>9.1f}%")
    print("\nVariaveis com utilidade baixa devem sair do espaco de features.")

    return df


# ---------------------------------------------------------------------------
# (B) Duplicatas
# ---------------------------------------------------------------------------

def fingerprint(fp: Path, cols):
    """Hash do conteudo numerico das colunas presentes."""
    try:
        df = pl.read_parquet(fp)
    except Exception:
        return None
    h = hashlib.sha256()
    for c in cols:
        if c in df.columns:
            a = df[c].to_numpy().astype(float)
            a = np.nan_to_num(a, nan=-9.99e30)
            h.update(c.encode())
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def duplicates(root: Path, mf: pl.DataFrame):
    sec("(B) INSTANCIAS DUPLICADAS ENTRE POCOS")

    real = mf.filter(pl.col("instance_type") == "real")

    # 1) candidatos por n_obs identico em pocos diferentes
    by_n = defaultdict(list)
    for r in real.select(["filepath", "well_id", "n_obs",
                          "event_class_dir"]).to_dicts():
        by_n[r["n_obs"]].append(r)

    cands = {n: rs for n, rs in by_n.items()
             if len({r["well_id"] for r in rs}) > 1}
    print(f"  contagens de n_obs compartilhadas por >1 poco: {len(cands)}")
    if not cands:
        print("  Nenhum candidato. Sem duplicatas entre pocos.")
        return

    total = sum(len(rs) for rs in cands.values())
    print(f"  instancias envolvidas: {total}")
    print("\n  Verificando conteudo por hash...\n")

    hashes = defaultdict(list)
    for n, rs in cands.items():
        for r in rs:
            fp = Path(r["filepath"])
            if not fp.exists():
                fp = root / str(r["event_class_dir"]) / fp.name
            if not fp.exists():
                continue
            hh = fingerprint(fp, SHARED)
            if hh:
                hashes[hh].append((r["well_id"], r["event_class_dir"], fp.name))

    dups = {h: v for h, v in hashes.items()
            if len({w for w, _, _ in v}) > 1}

    if not dups:
        print("  >>> Nenhuma duplicata confirmada. As coincidencias de n_obs")
        print("      sao fortuitas. Bom para a validade do LOWO.")
    else:
        print(f"  >>> {len(dups)} grupos de conteudo IDENTICO em pocos distintos:\n")
        for h, v in dups.items():
            print(f"    hash {h[:12]}...")
            for w, c, fn in v:
                print(f"      {w}  classe {c}  {fn}")
            print()
        print("  Isso e vazamento: leave-one-well-out nao isola os folds.")


def main():
    pos, out_dir = [], Path(".")
    for a in sys.argv[1:]:
        if a.startswith("--out-dir"):
            out_dir = Path(a.split("=", 1)[1])
        elif a.startswith("--"):
            print(f"Opcao desconhecida: {a}")
            sys.exit(1)
        else:
            pos.append(a)

    if not pos:
        print(__doc__)
        sys.exit(1)

    root = Path(pos[0]).expanduser().resolve()
    man = Path(pos[1]) if len(pos) > 1 else Path("manifest_3w.parquet")
    txp = Path(pos[2]) if len(pos) > 2 else Path("taxonomy_pairs.parquet")
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (man, txp):
        if not p.exists():
            print(f"ERRO: nao encontrado: {p}")
            sys.exit(1)

    mf = pl.read_parquet(man)
    tx = pl.read_parquet(txp)
    print(f"raiz: {root}\nmanifesto: {man}\ntaxonomia: {txp}")

    df = viability(mf, tx)
    dest = out_dir / "class9_viability.parquet"
    df.write_parquet(dest)
    print(f"\nSalvo: {dest}")

    duplicates(root, mf)
    print()


if __name__ == "__main__":
    main()