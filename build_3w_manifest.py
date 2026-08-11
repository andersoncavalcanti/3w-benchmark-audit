#!/usr/bin/env python3
"""
build_3w_manifest.py
====================
Passo 1 da Linha A: inventario/caracterizacao do 3W Dataset 2.0.0.

Produz um manifesto (1 linha por instancia) com:
  - metadados: tipo (real/simulated/drawn), classe do evento, ID do poco, duracao
  - por variavel: fracao ausente, fracao "estagnada" (diff==0), n valores distintos
  - composicao de rotulos em nivel de OBSERVACAO (class labels e state labels)

E imprime os tres "gates" que definem o desenho experimental:
  Gate 1 - disponibilidade de variaveis por dominio (real vs simulado)
  Gate 2 - viabilidade por classe (instancias e pocos distintos)
  Gate 3 - presenca de operacao normal (label 0) DENTRO das instancias simuladas

Uso:
    pip install polars pyarrow
    python build_3w_manifest.py /caminho/para/3W/dataset

Saidas: manifest_3w.parquet, manifest_3w.csv
"""

import sys
import re
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

# As 27 variaveis do 3W Dataset 2.0.0
VARIABLES = [
    "ABER-CKGL", "ABER-CKP",
    "ESTADO-DHSV", "ESTADO-M1", "ESTADO-M2", "ESTADO-PXO",
    "ESTADO-SDV-GL", "ESTADO-SDV-P", "ESTADO-W1", "ESTADO-W2", "ESTADO-XO",
    "P-ANULAR", "P-JUS-BS", "P-JUS-CKGL", "P-JUS-CKP",
    "P-MON-CKGL", "P-MON-CKP", "P-MON-SDV-P", "P-PDG",
    "PT-P", "P-TPT",
    "QBS", "QGL",
    "T-JUS-CKP", "T-MON-CKP", "T-PDG", "T-TPT",
]

# Variaveis discretas: ficar constante e comportamento legitimo, nao "congelamento".
DISCRETE_VARS = [v for v in VARIABLES if v.startswith("ESTADO-")]
CONTINUOUS_VARS = [v for v in VARIABLES if v not in DISCRETE_VARS]

# Codigos de rotulo de classe (estado permanente + condicao transiente)
CLASS_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
               101, 102, 105, 106, 107, 108, 109]
# Codigos de rotulo de estado operacional
STATE_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8]

EVENT_DIRS = [str(i) for i in range(10)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_filename(path: Path):
    """Extrai tipo de instancia e ID do poco a partir do nome do arquivo."""
    name = path.stem
    m = re.match(r"^WELL-(\d+)_(\d{14})$", name)
    if m:
        return "real", f"WELL-{m.group(1)}", m.group(2)
    if name.startswith("SIMULATED_"):
        return "simulated", None, None
    if name.startswith("DRAWN_"):
        return "drawn", None, None
    return "unknown", None, None


def find_label_columns(df: pl.DataFrame):
    """
    Identifica colunas de rotulo (tudo que nao e variavel nem timestamp).
    Robusto a mudancas de nomenclatura entre versoes ('class'/'state',
    'class_label'/'state_label', etc).
    """
    ts_cols = [c for c, dt in zip(df.columns, df.dtypes)
               if dt in (pl.Datetime, pl.Date)]
    label_cols = [c for c in df.columns
                  if c not in VARIABLES and c not in ts_cols
                  and not c.startswith("__index")]
    return ts_cols, label_cols


def classify_label_columns(label_cols):
    """Separa heuristicamente coluna de classe e coluna de estado."""
    class_col = state_col = None
    for c in label_cols:
        lc = c.lower()
        if "state" in lc or "estado" in lc:
            state_col = c
        elif "class" in lc or "classe" in lc or lc == "label":
            class_col = c
    # fallback: primeira coluna vira classe
    if class_col is None and label_cols:
        class_col = label_cols[0]
    return class_col, state_col


# ---------------------------------------------------------------------------
# Caracterizacao de uma instancia
# ---------------------------------------------------------------------------

def characterize(path: Path, event_dir: str) -> dict:
    df = pl.read_parquet(path)
    n = df.height

    inst_type, well_id, ts_str = parse_filename(path)
    ts_cols, label_cols = find_label_columns(df)
    class_col, state_col = classify_label_columns(label_cols)

    row = {
        "filepath": str(path),
        "filename": path.name,
        "event_class_dir": int(event_dir),
        "instance_type": inst_type,
        "well_id": well_id,
        "file_timestamp": ts_str,
        "n_obs": n,
        "n_columns": df.width,
        "label_cols_found": "|".join(label_cols),
    }

    # Duracao e cadencia temporal
    if ts_cols:
        tcol = ts_cols[0]
        t = df[tcol]
        row["first_ts"] = str(t.min())
        row["last_ts"] = str(t.max())
        if n > 1:
            deltas = t.diff().drop_nulls()
            try:
                row["median_dt_s"] = float(
                    deltas.dt.total_seconds().median()
                )
            except Exception:
                row["median_dt_s"] = None
            row["duration_s"] = (
                (t.max() - t.min()).total_seconds()
                if t.max() is not None else None
            )
    else:
        row["first_ts"] = row["last_ts"] = None
        row["median_dt_s"] = row["duration_s"] = None

    # ---- Estatisticas por variavel ----
    for v in VARIABLES:
        if v not in df.columns:
            row[f"{v}__present"] = 0
            row[f"{v}__missing_frac"] = 1.0
            row[f"{v}__flat_frac"] = None
            row[f"{v}__n_unique"] = 0
            continue

        s = df[v]
        n_null = s.null_count()
        miss = n_null / n if n else 1.0
        row[f"{v}__present"] = int(miss < 1.0)
        row[f"{v}__missing_frac"] = float(miss)
        row[f"{v}__n_unique"] = int(s.n_unique())

        # "flat_frac": fracao de amostras consecutivas identicas.
        # Proxy barato de congelamento. So faz sentido para variaveis continuas.
        if v in CONTINUOUS_VARS and n > 1 and miss < 1.0:
            d = s.diff().drop_nulls()
            row[f"{v}__flat_frac"] = (
                float((d == 0).sum() / d.len()) if d.len() else None
            )
        else:
            row[f"{v}__flat_frac"] = None

    # ---- Composicao de rotulos em nivel de observacao ----
    if class_col and class_col in df.columns:
        cs = df[class_col]
        row["class_null_frac"] = float(cs.null_count() / n) if n else None
        for code in CLASS_CODES:
            row[f"class_{code}_frac"] = float((cs == code).sum() / n) if n else 0.0
    else:
        row["class_null_frac"] = None
        for code in CLASS_CODES:
            row[f"class_{code}_frac"] = None

    if state_col and state_col in df.columns:
        ss = df[state_col]
        row["state_null_frac"] = float(ss.null_count() / n) if n else None
        for code in STATE_CODES:
            row[f"state_{code}_frac"] = float((ss == code).sum() / n) if n else 0.0
    else:
        row["state_null_frac"] = None
        for code in STATE_CODES:
            row[f"state_{code}_frac"] = None

    return row


# ---------------------------------------------------------------------------
# Relatorio dos tres gates
# ---------------------------------------------------------------------------

def report_gates(mf: pl.DataFrame):
    line = "=" * 78

    print(f"\n{line}\nGATE 1 - DISPONIBILIDADE DE VARIAVEIS POR DOMINIO\n{line}")
    print("presence_rate = fracao de instancias em que a variavel tem algum dado.")
    print("A INTERSECAO real x simulado define o espaco de features do estudo.\n")
    print(f"{'variavel':<16}{'real':>10}{'simulado':>12}{'desenhado':>12}  interseccao")
    print("-" * 78)
    usable = []
    for v in VARIABLES:
        col = f"{v}__present"
        rates = {}
        for t in ("real", "simulated", "drawn"):
            sub = mf.filter(pl.col("instance_type") == t)
            rates[t] = float(sub[col].mean()) if sub.height else 0.0
        ok = rates["real"] > 0.5 and rates["simulated"] > 0.5
        if ok:
            usable.append(v)
        print(f"{v:<16}{rates['real']:>10.2f}{rates['simulated']:>12.2f}"
              f"{rates['drawn']:>12.2f}  {'SIM' if ok else '--'}")
    print(f"\n>>> {len(usable)} variaveis utilizaveis em sim->real:")
    print("   ", ", ".join(usable) if usable else "(NENHUMA)")

    print(f"\n{line}\nGATE 2 - VIABILIDADE POR CLASSE\n{line}")
    print(f"{'classe':>7}{'real':>8}{'simul':>8}{'desen':>8}{'pocos':>8}   LOWO viavel?")
    print("-" * 78)
    for c in range(10):
        sub = mf.filter(pl.col("event_class_dir") == c)
        n_real = sub.filter(pl.col("instance_type") == "real").height
        n_sim = sub.filter(pl.col("instance_type") == "simulated").height
        n_dr = sub.filter(pl.col("instance_type") == "drawn").height
        wells = sub.filter(pl.col("well_id").is_not_null())["well_id"].n_unique()
        lowo = "sim" if wells >= 5 else ("marginal" if wells >= 3 else "NAO")
        s2r = "sim" if (n_real >= 10 and n_sim >= 10) else "NAO"
        print(f"{c:>7}{n_real:>8}{n_sim:>8}{n_dr:>8}{wells:>8}   "
              f"LOWO={lowo:<9} sim->real={s2r}")

    print(f"\n{line}\nGATE 3 - OPERACAO NORMAL DENTRO DAS INSTANCIAS SIMULADAS\n{line}")
    print("Pergunta critica: instancias simuladas contem observacoes rotuladas 0?")
    print("Se NAO, um modelo treinado so em simulado nao aprende normalidade.\n")
    sim = mf.filter(pl.col("instance_type") == "simulated")
    if sim.height and sim["class_0_frac"].null_count() < sim.height:
        with_normal = sim.filter(pl.col("class_0_frac") > 0).height
        mean_frac = float(sim["class_0_frac"].mean())
        print(f"  instancias simuladas totais ............ {sim.height}")
        print(f"  com alguma observacao classe 0 ......... {with_normal} "
              f"({100*with_normal/sim.height:.1f}%)")
        print(f"  fracao media de obs. classe 0 .......... {mean_frac:.3f}")
        if with_normal == 0:
            print("\n  >>> ACHADO FORTE: nenhuma normalidade simulada.")
            print("      O desenho experimental precisa ser reformulado.")
        else:
            print("\n  >>> Normalidade simulada existe em nivel de observacao.")
            print("      Treino sim->real e viavel; construir janelas por rotulo.")
    else:
        print("  Coluna de classe nao localizada. Verifique 'label_cols_found'.")

    print(f"\n{line}\nQUALIDADE DO DADO REAL\n{line}")
    real = mf.filter(pl.col("instance_type") == "real")
    if real.height:
        miss = [float(real[f"{v}__missing_frac"].mean()) for v in VARIABLES]
        print(f"  fracao media de variaveis ausentes ..... "
              f"{sum(miss)/len(miss):.3f}")
        flats = [(v, float(real[f"{v}__flat_frac"].mean()))
                 for v in CONTINUOUS_VARS
                 if real[f"{v}__flat_frac"].null_count() < real.height]
        flats.sort(key=lambda x: -x[1])
        print("  variaveis continuas mais estagnadas (proxy de congelamento):")
        for v, f in flats[:8]:
            print(f"    {v:<16} {f:.3f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        print(f"ERRO: diretorio nao encontrado: {root}")
        sys.exit(1)

    files = []
    for d in EVENT_DIRS:
        sub = root / d
        if sub.is_dir():
            files += [(p, d) for p in sorted(sub.glob("*.parquet"))]

    if not files:
        print(f"ERRO: nenhum .parquet em {root}/[0-9]/")
        print("Aponte para a raiz do dataset (a que contem dataset.ini).")
        sys.exit(1)

    print(f"Encontrados {len(files)} arquivos. Processando...")
    rows, failures = [], []
    for i, (p, d) in enumerate(files, 1):
        try:
            rows.append(characterize(p, d))
        except Exception as e:
            failures.append((str(p), repr(e)))
        if i % 200 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    if failures:
        print(f"\nAVISO: {len(failures)} arquivos falharam. Primeiros 5:")
        for f, e in failures[:5]:
            print(f"  {f}\n    {e}")

    mf = pl.DataFrame(rows, infer_schema_length=None)
    mf.write_parquet("manifest_3w.parquet")
    mf.write_csv("manifest_3w.csv")
    print(f"\nManifesto salvo: manifest_3w.parquet / .csv "
          f"({mf.height} linhas x {mf.width} colunas)")

    report_gates(mf)


if __name__ == "__main__":
    main()