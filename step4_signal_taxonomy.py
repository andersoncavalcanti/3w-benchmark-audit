#!/usr/bin/env python3
"""
step4_signal_taxonomy.py  (versao corrigida)
============================================
Corrige a analise (2) do passo 3.

Erro corrigido na analise: as estatisticas de corrida eram agregadas ENTRE
instancias, de modo que uma unica instancia patologica dominava o resultado;
e faltava a categoria CONSTANTE (variavel nao-nula mas invariante na
instancia inteira), que passa em qualquer verificacao de faltantes e carrega
zero informacao.

Classifica cada par (instancia, variavel) em cinco categorias:

  AUSENTE     - integralmente nula
  CONSTANTE   - <=2 valores distintos, ou uma unica corrida cobrindo >50%
                das observacoes. Tag presente porem sem informacao.
  CONGELADA   - varia, mas >5% das observacoes presas em corridas > 10 min
  QUANTIZADA  - corridas curtas; diferencas sao multiplos de um passo discreto
  CONTINUA    - corridas curtas; sem passo discreto detectavel

Uso:
    python step4_signal_taxonomy.py <raiz_dataset> [manifesto] [opcoes]

Opcoes:
    --max-files=N      processa apenas as N primeiras instancias (teste rapido)
    --out-dir=CAMINHO  diretorio de saida (padrao: diretorio atual)

Exemplos:
    python step4_signal_taxonomy.py C:\\data\\3W manifest_3w.parquet
    python step4_signal_taxonomy.py C:\\data\\3W manifest_3w.parquet --max-files=200
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

CONT_VARS = [
    "P-MON-CKP", "P-PDG", "P-TPT", "T-JUS-CKP", "T-TPT",
    "P-ANULAR", "P-JUS-CKGL", "P-JUS-CKP", "QGL", "T-PDG",
    "ABER-CKP", "ABER-CKGL", "T-MON-CKP",
]

SHARED = ["P-MON-CKP", "P-PDG", "P-TPT", "T-JUS-CKP", "T-TPT"]

FS_HZ = 1.0
FREEZE_S = 600           # corrida "longa" = 10 min
FREEZE_FRAC = 0.05       # >5% das obs em corridas longas => congelada
CONST_DOMINANCE = 0.50   # uma corrida cobrindo >50% => constante
QUANT_TOL = 1e-6
QUANT_FRAC = 0.90        # >=90% das diferencas em multiplos de q => quantizada

CATS = ["AUSENTE", "CONSTANTE", "CONGELADA", "QUANTIZADA", "CONTINUA"]
LINE = "=" * 78


def sec(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


# ---------------------------------------------------------------------------
# Leitura de esquema robusta a versao do polars
# ---------------------------------------------------------------------------

def parquet_columns(fp: Path):
    """Nomes das colunas sem ler os dados. Tenta varias APIs do polars."""
    try:
        return list(pl.read_parquet_schema(fp).keys())
    except Exception:
        pass
    try:
        return list(pl.scan_parquet(fp).collect_schema().names())
    except Exception:
        pass
    try:
        return list(pl.scan_parquet(fp).columns)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Classificacao de uma serie
# ---------------------------------------------------------------------------

def classify(x: np.ndarray):
    """Classifica uma serie. Retorna (categoria, dict de metricas)."""
    n = x.size
    if n == 0 or np.all(np.isnan(x)):
        return "AUSENTE", {}

    n_valid = int((~np.isnan(x)).sum())
    if n_valid == 0:
        return "AUSENTE", {}

    # Corridas de valores identicos (NaN quebra a corrida e depois e filtrado)
    starts = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    ok = ~np.isnan(values)
    lengths, values = lengths[ok], values[ok]
    if lengths.size == 0:
        return "AUSENTE", {}

    obs_in_runs = int(lengths.sum())
    longest = int(lengths.max())
    n_unique = int(np.unique(values).size)
    dominance = longest / obs_in_runs

    m = {
        "n_valid": n_valid,
        "n_unique": n_unique,
        "n_runs": int(lengths.size),
        "longest_run": longest,
        "dominance": float(dominance),
        "median_run": float(np.median(lengths)),
        "p99_run": float(np.percentile(lengths, 99)),
        "q_est": None,
        "frac_long": None,
        "quant_hit": None,
    }

    thr = FREEZE_S * FS_HZ
    frac_long = float(lengths[lengths > thr].sum() / obs_in_runs)
    m["frac_long"] = frac_long

    # 1) Constante: tag presente, sem informacao
    if n_unique <= 2 or dominance > CONST_DOMINANCE:
        return "CONSTANTE", m

    # Passo de quantizacao estimado
    d = np.abs(np.diff(values))
    d = d[np.isfinite(d) & (d > 0)]
    q = float(np.min(d)) if d.size else None
    m["q_est"] = q

    # 2) Congelada
    if frac_long > FREEZE_FRAC:
        return "CONGELADA", m

    # 3) Quantizada vs continua
    if q is not None and d.size >= 10:
        ratios = d / q
        hits = np.abs(ratios - np.round(ratios)) < QUANT_TOL
        m["quant_hit"] = float(hits.mean())
        if hits.mean() >= QUANT_FRAC:
            return "QUANTIZADA", m

    return "CONTINUA", m


# ---------------------------------------------------------------------------
# Relatorios
# ---------------------------------------------------------------------------

def report(tx: pl.DataFrame):
    sec("TAXONOMIA POR DOMINIO (% dos pares instancia x variavel)")
    for dom in ("real", "simulated", "drawn"):
        sub = tx.filter(pl.col("instance_type") == dom)
        if not sub.height:
            continue
        print(f"--- {dom} ({sub.height} pares) ---")
        for c in CATS:
            n = sub.filter(pl.col("category") == c).height
            pct = 100 * n / sub.height
            print(f"  {c:<12}{n:>7}  {pct:>6.2f}%  {'#' * int(pct / 2)}")
        print()

    sec("TAXONOMIA POR VARIAVEL (dominio real)")
    hdr = f"{'variavel':<14}" + "".join(f"{c[:9]:>11}" for c in CATS)
    print(hdr)
    print("-" * len(hdr))
    real = tx.filter(pl.col("instance_type") == "real")
    for v in CONT_VARS:
        s = real.filter(pl.col("variable") == v)
        if not s.height:
            continue
        cells = "".join(
            f"{100 * s.filter(pl.col('category') == c).height / s.height:>11.1f}"
            for c in CATS)
        print(f"{v:<14}{cells}")
    print("\n(valores em % das instancias reais)")

    sec("IMPACTO: TAGS PRESENTES POREM CONSTANTES")
    nc = real.filter(pl.col("category") == "CONSTANTE").height
    npres = real.filter(pl.col("category") != "AUSENTE").height
    print(f"  pares nao-ausentes no real ............ {npres}")
    print(f"  destes, CONSTANTES (sem informacao) ... {nc} "
          f"({100 * nc / npres if npres else 0:.2f}%)")
    print("\n  A contagem de variaveis 'disponiveis' baseada apenas em")
    print("  nulidade e otimista nessa proporcao.")

    sec("AS 5 VARIAVEIS COMPARTILHADAS: QUANTAS SOBREVIVEM?")
    print(f"{'variavel':<14}{'util (%)':>12}{'constante':>12}"
          f"{'congelada':>12}{'ausente':>11}")
    print("-" * 61)
    for v in SHARED:
        s = real.filter(pl.col("variable") == v)
        if not s.height:
            continue
        p = {c: 100 * s.filter(pl.col("category") == c).height / s.height
             for c in CATS}
        util = p["QUANTIZADA"] + p["CONTINUA"]
        print(f"{v:<14}{util:>12.1f}{p['CONSTANTE']:>12.1f}"
              f"{p['CONGELADA']:>12.1f}{p['AUSENTE']:>11.1f}")
    print("\n'util' = quantizada + continua (carrega informacao aproveitavel)")

    sec("MAPA POCO x VARIAVEL (categoria modal, instancias reais)")
    tab = defaultdict(dict)
    wells_all = [w for w in real["well_id"].unique().to_list() if w is not None]
    for v in CONT_VARS:
        sv = real.filter(pl.col("variable") == v)
        for w in wells_all:
            s = sv.filter(pl.col("well_id") == w)
            if not s.height:
                continue
            counts = {c: s.filter(pl.col("category") == c).height for c in CATS}
            tab[w][v] = max(counts, key=counts.get)

    sym = {"AUSENTE": ".", "CONSTANTE": "K", "CONGELADA": "F",
           "QUANTIZADA": "Q", "CONTINUA": "C"}
    print("legenda: . ausente | K constante | F congelada | "
          "Q quantizada | C continua\n")
    print(f"{'poco':<14}" + "".join(f"{v[:9]:>11}" for v in CONT_VARS))
    print("-" * (14 + 11 * len(CONT_VARS)))
    signatures = defaultdict(list)
    for w in sorted(tab.keys()):
        sig = tuple(tab[w].get(v, "AUSENTE") for v in CONT_VARS)
        signatures[sig].append(w)
        print(f"{w:<14}"
              + "".join(f"{sym.get(tab[w].get(v, 'AUSENTE'), '?'):>11}"
                        for v in CONT_VARS))

    print(f"\nAssinaturas de instrumentacao distintas: {len(signatures)} "
          f"para {len(tab)} pocos")
    dup = {s: ws for s, ws in signatures.items() if len(ws) > 1}
    if dup:
        print("Pocos que compartilham exatamente a mesma assinatura:")
        for ws in dup.values():
            print(f"  {ws}")
    else:
        print(">>> Cada poco tem assinatura unica: a mascara identifica")
        print("    o POCO individualmente, nao apenas o dominio.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pos, max_files, out_dir = [], None, Path(".")
    for a in sys.argv[1:]:
        if a.startswith("--max-files"):
            try:
                max_files = int(a.split("=", 1)[1])
            except (IndexError, ValueError):
                print("Use --max-files=N")
                sys.exit(1)
        elif a.startswith("--out-dir"):
            try:
                out_dir = Path(a.split("=", 1)[1])
            except IndexError:
                print("Use --out-dir=CAMINHO")
                sys.exit(1)
        elif a.startswith("--"):
            print(f"Opcao desconhecida: {a}")
            sys.exit(1)
        else:
            pos.append(a)

    if not pos:
        print(__doc__)
        sys.exit(1)

    root = Path(pos[0]).expanduser().resolve()
    man = Path(pos[1]).expanduser() if len(pos) > 1 else Path("manifest_3w.parquet")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        print(f"ERRO: raiz nao encontrada: {root}")
        sys.exit(1)
    if not man.exists():
        print(f"ERRO: manifesto nao encontrado: {man}")
        sys.exit(1)

    mf = pl.read_parquet(man)
    rows = mf.select(["filepath", "instance_type", "event_class_dir",
                      "well_id", "n_obs"]).to_dicts()
    if max_files:
        rows = rows[:max_files]

    print(f"raiz ......: {root}")
    print(f"manifesto .: {man}")
    print(f"saida .....: {out_dir.resolve()}")
    print(f"Classificando {len(rows)} instancias x {len(CONT_VARS)} variaveis...\n")

    out, missing_files = [], 0
    for i, r in enumerate(rows, 1):
        fp = Path(r["filepath"])
        if not fp.exists():
            fp = root / str(r["event_class_dir"]) / fp.name
        if not fp.exists():
            missing_files += 1
            continue

        cols = parquet_columns(fp)
        base = {k: r[k] for k in ("instance_type", "event_class_dir", "well_id")}

        df = None
        if cols is not None:
            want = [c for c in CONT_VARS if c in cols]
            try:
                df = pl.read_parquet(fp, columns=want) if want else pl.DataFrame()
            except Exception:
                df = None
        if df is None:
            try:
                df = pl.read_parquet(fp)
            except Exception:
                missing_files += 1
                continue

        for v in CONT_VARS:
            if v not in df.columns:
                out.append({**base, "filename": fp.name,
                            "variable": v, "category": "AUSENTE"})
                continue
            cat, m = classify(df[v].to_numpy().astype(float))
            out.append({**base, "filename": fp.name,
                        "variable": v, "category": cat, **m})

        if i % 200 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    if missing_files:
        print(f"\nAVISO: {missing_files} arquivos nao localizados/ilegiveis.")

    tx = pl.DataFrame(out, infer_schema_length=None)
    dest = out_dir / "taxonomy_pairs.parquet"
    tx.write_parquet(dest)
    print(f"\nSalvo: {dest}  ({tx.height} pares)")

    report(tx)


if __name__ == "__main__":
    main()