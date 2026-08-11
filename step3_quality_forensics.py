#!/usr/bin/env python3
"""
step3_quality_forensics.py
==========================
Passo 3 da Linha A. Fecha a caracterizacao com tres analises:

  (1) BINARIDADE DOS FALTANTES  [roda sobre o manifesto]
      Histograma de missing_frac sobre todos os pares (instancia, variavel).
      Hipotese: a massa se concentra em exatamente 0.0 e exatamente 1.0,
      o que caracteriza INDISPONIBILIDADE DE TAG e nao falha de sensor.

  (2) CONGELAMENTO vs QUANTIZACAO  [rele os parquet brutos, por amostragem]
      Distribuicao de comprimentos de corrida constante por variavel.
      Quantizacao com passo q e inclinacao |ds/dt| gera corridas de
      comprimento ~ q/|ds/dt| (curtas, cauda leve).
      Congelamento gera cauda pesada: corridas de minutos a horas.

  (3) TABELA CORRIGIDA DA CLASSE 9  [corrige o bug de `or` do passo 2]

Uso:
    python step3_quality_forensics.py <raiz_dataset> [manifest_3w.parquet]
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

SHARED = ["P-MON-CKP", "P-PDG", "P-TPT", "T-JUS-CKP", "T-TPT"]
EXTRA = ["P-ANULAR", "P-JUS-CKGL", "QGL", "T-PDG", "ABER-CKP"]
CONT_VARS = SHARED + EXTRA

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

MAX_FILES_PER_GROUP = 12   # amostragem para (2)
FS_HZ = 1.0                # cadencia nominal do 3W
LINE = "=" * 78


def sec(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


# ---------------------------------------------------------------------------
# (1) Binaridade dos faltantes
# ---------------------------------------------------------------------------

def missingness_histogram(mf):
    sec("(1) BINARIDADE DOS FALTANTES")
    print("Pares (instancia, variavel) por faixa de fracao de faltantes.\n")

    bins = [
        ("exatamente 0.0     ", lambda a: a == 0.0),
        ("(0.0 , 0.01]       ", lambda a: (a > 0.0) & (a <= 0.01)),
        ("(0.01, 0.10]       ", lambda a: (a > 0.01) & (a <= 0.10)),
        ("(0.10, 0.50]       ", lambda a: (a > 0.10) & (a <= 0.50)),
        ("(0.50, 0.99]       ", lambda a: (a > 0.50) & (a <= 0.99)),
        ("(0.99, 1.0)        ", lambda a: (a > 0.99) & (a < 1.0)),
        ("exatamente 1.0     ", lambda a: a == 1.0),
    ]

    for dom in ("real", "simulated"):
        sub = mf.filter(pl.col("instance_type") == dom)
        if not sub.height:
            continue
        vals = np.concatenate([
            sub[f"{v}__missing_frac"].fill_null(1.0).to_numpy().astype(float)
            for v in VARIABLES
        ])
        tot = vals.size
        print(f"--- dominio: {dom}  ({sub.height} instancias x "
              f"{len(VARIABLES)} variaveis = {tot} pares) ---")
        extremes = 0
        for label, fn in bins:
            n = int(fn(vals).sum())
            pct = 100 * n / tot
            bar = "#" * int(pct / 2)
            print(f"  {label} {n:>8}  {pct:>6.2f}%  {bar}")
            if label.startswith("exatamente"):
                extremes += n
        print(f"\n  >>> massa nos extremos (0.0 ou 1.0): "
              f"{100*extremes/tot:.2f}%\n")

    print("Leitura: massa concentrada nos extremos => a variavel esta")
    print("integralmente presente ou integralmente ausente na instancia.")
    print("Isso e INDISPONIBILIDADE DE TAG, nao dropout de sensor, e torna")
    print("inadequada a aplicacao de imputacao temporal (GRU-D, SAITS etc).")


# ---------------------------------------------------------------------------
# (2) Congelamento vs quantizacao
# ---------------------------------------------------------------------------

def constant_runs(x: np.ndarray):
    """Comprimentos das corridas de valores identicos, ignorando NaN."""
    if x.size == 0:
        return np.array([]), np.array([])
    starts = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
    lengths = np.diff(np.r_[starts, x.size])
    values = x[starts]
    ok = ~np.isnan(values)
    return lengths[ok], values[ok]


def quantization_step(values: np.ndarray):
    """Passo de quantizacao estimado: menor diferenca positiva entre
    valores consecutivos distintos."""
    if values.size < 3:
        return None
    d = np.abs(np.diff(values))
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return None
    return float(np.min(d))


def freezing_analysis(root: Path, mf):
    sec("(2) CONGELAMENTO vs QUANTIZACAO")
    print(f"Amostra de ate {MAX_FILES_PER_GROUP} instancias por dominio.")
    print(f"Cadencia assumida: {FS_HZ:.0f} Hz.\n")

    for dom, tipo in (("real", "real"), ("simulado", "simulated")):
        sub = mf.filter(pl.col("instance_type") == tipo)
        if not sub.height:
            continue
        paths = sub["filepath"].to_list()
        step = max(1, len(paths) // MAX_FILES_PER_GROUP)
        paths = paths[::step][:MAX_FILES_PER_GROUP]

        acc = {v: [] for v in CONT_VARS}
        qsteps = {v: [] for v in CONT_VARS}
        for p in paths:
            fp = Path(p)
            if not fp.exists():
                fp = root / Path(p).parent.name / Path(p).name
            if not fp.exists():
                continue
            try:
                df = pl.read_parquet(fp)
            except Exception:
                continue
            for v in CONT_VARS:
                if v not in df.columns:
                    continue
                x = df[v].to_numpy().astype(float)
                if np.all(np.isnan(x)):
                    continue
                lens, vals = constant_runs(x)
                if lens.size:
                    acc[v].append(lens)
                    q = quantization_step(vals)
                    if q is not None:
                        qsteps[v].append(q)

        print(f"--- dominio: {dom} ---")
        hdr = (f"{'variavel':<14}{'q_est':>10}{'med':>7}{'p90':>8}"
               f"{'p99':>9}{'max':>10}{'%>60s':>8}{'%>10min':>9}  veredito")
        print(hdr)
        print("-" * len(hdr))
        for v in CONT_VARS:
            if not acc[v]:
                continue
            L = np.concatenate(acc[v]).astype(float)
            total_obs = L.sum()
            q = np.median(qsteps[v]) if qsteps[v] else float("nan")
            med, p90, p99, mx = (np.median(L), np.percentile(L, 90),
                                 np.percentile(L, 99), L.max())
            thr1, thr2 = 60 * FS_HZ, 600 * FS_HZ
            f60 = 100 * L[L > thr1].sum() / total_obs
            f600 = 100 * L[L > thr2].sum() / total_obs

            # Veredito: cauda pesada em relacao a mediana => congelamento
            ratio = p99 / med if med > 0 else np.inf
            if f600 > 5:
                verd = "CONGELAMENTO"
            elif ratio > 50 and f60 > 5:
                verd = "misto"
            else:
                verd = "quantizacao"
            print(f"{v:<14}{q:>10.4g}{med:>7.0f}{p90:>8.0f}{p99:>9.0f}"
                  f"{mx:>10.0f}{f60:>8.1f}{f600:>9.1f}  {verd}")
        print()

    print("q_est  : passo de quantizacao estimado (menor variacao positiva)")
    print("med/p90/p99/max : comprimento das corridas constantes, em amostras")
    print("%>60s  : fracao de OBSERVACOES dentro de corridas maiores que 60 s")
    print("\nQuantizacao => corridas curtas, cauda leve, p99/med moderado.")
    print("Congelamento => cauda pesada; observacoes presas por >10 min.")


# ---------------------------------------------------------------------------
# (3) Tabela corrigida da classe 9
# ---------------------------------------------------------------------------

def class9_fixed(mf):
    sec("(3) CLASSE 9 - INSTANCIAS REAIS (tabela corrigida)")

    def cov(v):
        # correcao do bug: 0.0 é valor valido, nao "ausente"
        m = v
        return 1.0 - (1.0 if m is None else float(m))

    sub = (mf.filter((pl.col("event_class_dir") == 9)
                     & (pl.col("instance_type") == "real"))
             .sort(["well_id", "n_obs"]))

    hdr = (f"{'poco':<13}{'n_obs':>9}{'dur_h':>7}"
           f"{'%c0':>7}{'%c109':>7}{'%c9':>7}"
           + "".join(f"{v[:9]:>10}" for v in SHARED) + "  5/5")
    print(hdr)
    print("-" * len(hdr))

    n_full = 0
    wells_full = set()
    for r in sub.to_dicts():
        dur = (r["duration_s"] or 0) / 3600
        f0 = (r.get("class_0_frac") or 0) * 100
        f109 = (r.get("class_109_frac") or 0) * 100
        f9 = (r.get("class_9_frac") or 0) * 100
        covs = [cov(r[f"{v}__missing_frac"]) for v in SHARED]
        full = all(c >= 0.99 for c in covs)
        if full:
            n_full += 1
            wells_full.add(r["well_id"])
        print(f"{str(r['well_id']):<13}{r['n_obs']:>9}{dur:>7.1f}"
              f"{f0:>7.1f}{f109:>7.1f}{f9:>7.1f}"
              + "".join(f"{c:>10.2f}" for c in covs)
              + ("  SIM" if full else "   --"))

    print("-" * len(hdr))
    print(f"\nInstancias com as 5 variaveis completas: {n_full} "
          f"em {len(wells_full)} pocos")
    print(f"Pocos: {sorted(wells_full)}")

    # Orcamento de normalidade intra-instancia
    tot_c0 = sum(int((r.get("class_0_frac") or 0) * r["n_obs"])
                 for r in sub.to_dicts())
    tot_c109 = sum(int((r.get("class_109_frac") or 0) * r["n_obs"])
                   for r in sub.to_dicts())
    tot_c9 = sum(int((r.get("class_9_frac") or 0) * r["n_obs"])
                 for r in sub.to_dicts())
    print(f"\nOrcamento de observacoes (todas as instancias reais classe 9):")
    print(f"  normal  (rotulo 0)   : {tot_c0:>10}  ({tot_c0/3600:.1f} h)")
    print(f"  transiente (rot 109) : {tot_c109:>10}  ({tot_c109/3600:.1f} h)")
    print(f"  permanente (rot 9)   : {tot_c9:>10}  ({tot_c9/3600:.1f} h)")
    print("\nA normalidade da deteccao binaria sai do rotulo 0 INTRA-instancia,")
    print("ja que nenhum poco com classe 9 possui instancia de classe 0.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1]).expanduser().resolve()
    man = sys.argv[2] if len(sys.argv) > 2 else "manifest_3w.parquet"
    mf = pl.read_parquet(man)
    print(f"Manifesto: {mf.height} instancias | raiz: {root}")

    missingness_histogram(mf)
    freezing_analysis(root, mf)
    class9_fixed(mf)
    print()


if __name__ == "__main__":
    main()