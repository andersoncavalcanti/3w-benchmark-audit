#!/usr/bin/env python3
"""
step10_interpolation_test.py
============================
Testa a hipotese de Melo et al. (2022, Comput. Chem. Eng. 165:107964):
a taxa nominal de 1 Hz do 3W nao corresponde a medicoes reais; o sistema PI
interpola linearmente entre medicoes efetivas.

METODO (deterministico, nao heuristico)
  Em uma rampa linear, a primeira diferenca d1 e constante e a segunda
  diferenca d2 e zero. Logo:
    - medicao nativa a 1 Hz -> d1 muda a cada amostra, d2 != 0
    - interpolacao          -> d1 constante entre medicoes, d2 ~ 0
    - congelamento          -> caso particular com d1 constante = 0
  Os pontos com d2 != 0 sao os instantes de medicao reais; o espacamento
  entre eles e o INTERVALO EFETIVO DE AMOSTRAGEM.

  Via independente: autocorrelacao de |d2|. Interpolacao com periodo k
  produz pico de autocorrelacao no lag k. As duas estimativas devem
  concordar.

CONSEQUENCIAS AVALIADAS
  (1) fracao do sinal que e interpolacao, por dominio
  (2) intervalo efetivo por poco  -> a interpolacao tambem identifica o poco?
  (3) impacto na classe 9         -> quantas medicoes reais ha por janela?
  (4) o passo de quantizacao muda se calculado so nos pontos de medicao?

Uso:
    python step10_interpolation_test.py <raiz> [manifesto] [--sample=40]
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

SHARED = ["P-MON-CKP", "P-PDG", "P-TPT", "T-JUS-CKP", "T-TPT"]
MAX_LAG = 600            # lags para a autocorrelacao (10 min a 1 Hz)
MIN_LEN = 3000           # comprimento minimo de segmento valido
LINE = "=" * 78


def sec(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


# ---------------------------------------------------------------------------
# Nucleo do teste
# ---------------------------------------------------------------------------

def longest_valid(x):
    """Maior segmento contiguo sem NaN."""
    ok = np.isfinite(x)
    if not ok.any():
        return None
    d = np.diff(np.r_[0, ok.view(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    i = np.argmax(ends - starts)
    return x[starts[i]:ends[i]]


def acf_peak(y, max_lag):
    """Lag de maior autocorrelacao (>=2) e o valor do pico."""
    y = y - y.mean()
    n = len(y)
    if n < 4 * max_lag or np.allclose(y, 0):
        return None, None
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[:max_lag + 1]
    if ac[0] <= 0:
        return None, None
    ac = ac / ac[0]
    lag = int(np.argmax(ac[2:max_lag + 1])) + 2
    return lag, float(ac[lag])


def analyze(x):
    """Retorna metricas de interpolacao para uma serie."""
    seg = longest_valid(np.asarray(x, dtype=np.float64))
    if seg is None or seg.size < MIN_LEN:
        return None

    d1 = np.diff(seg)
    d2 = np.diff(d1)
    n2 = d2.size
    if n2 < 100:
        return None

    # quantum de armazenamento: menor variacao positiva observada
    pos = np.abs(d1[d1 != 0])
    q = float(pos.min()) if pos.size else 0.0

    # --- criterio estrito: d2 exatamente zero ---
    strict = float(np.mean(d2 == 0.0))

    # --- criterio tolerante: |d2| dentro do quantum de arredondamento ---
    tol = max(q, np.finfo(np.float64).eps * np.abs(seg).max())
    interp_mask = np.abs(d2) <= tol
    tolerant = float(np.mean(interp_mask))

    # --- intervalo efetivo: espacamento entre pontos com d2 significativo ---
    brk = np.flatnonzero(~interp_mask)
    if brk.size >= 10:
        gaps = np.diff(brk)
        gaps = gaps[gaps > 0]
        eff = float(np.median(gaps)) if gaps.size else 1.0
        eff_mode = Counter(gaps.tolist()).most_common(1)[0][0] if gaps.size else 1
    else:
        eff, eff_mode = float("nan"), -1

    # --- via independente: periodicidade de |d2| ---
    lag, peak = acf_peak(np.abs(d2), min(MAX_LAG, n2 // 4))

    # --- quantizacao: todos os pontos vs so os de medicao ---
    q_all = q
    if brk.size >= 10:
        vals = seg[np.r_[brk, brk[-1] + 2]]
        dv = np.abs(np.diff(vals))
        dv = dv[dv > 0]
        q_brk = float(dv.min()) if dv.size else float("nan")
    else:
        q_brk = float("nan")

    # fracao congelada (d1 == 0) separada da interpolacao (d1 constante != 0)
    frozen = float(np.mean(d1 == 0.0))

    return {
        "n": int(seg.size),
        "strict_interp": strict,
        "tolerant_interp": tolerant,
        "frozen": frozen,
        "eff_interval_s": eff,
        "eff_mode_s": eff_mode,
        "acf_lag": lag if lag else -1,
        "acf_peak": peak if peak else float("nan"),
        "q_all": q_all,
        "q_breakpoints": q_brk,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pos, sample_n = [], 40
    for a in sys.argv[1:]:
        if a.startswith("--sample"):
            sample_n = int(a.split("=", 1)[1])
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
    if not man.exists():
        print(f"ERRO: manifesto nao encontrado: {man}")
        sys.exit(1)

    mf = pl.read_parquet(man)

    # todas as instancias reais de classe 9 + amostra de outras + simuladas
    c9 = mf.filter((pl.col("event_class_dir") == 9)
                   & (pl.col("instance_type") == "real"))
    others = mf.filter((pl.col("event_class_dir") != 9)
                       & (pl.col("instance_type") == "real"))
    sims = mf.filter(pl.col("instance_type") == "simulated")

    def take(df, n):
        if df.height <= n:
            return df.to_dicts()
        step = max(1, df.height // n)
        return df.to_dicts()[::step][:n]

    targets = ([{**r, "_grp": "real_c9"} for r in c9.to_dicts()]
               + [{**r, "_grp": "real_outros"} for r in take(others, sample_n)]
               + [{**r, "_grp": "simulado"} for r in take(sims, sample_n)])

    print(f"Analisando {len(targets)} instancias x {len(SHARED)} variaveis\n")

    rows = []
    for i, r in enumerate(targets, 1):
        fp = Path(r["filepath"])
        if not fp.exists():
            fp = root / str(r["event_class_dir"]) / fp.name
        if not fp.exists():
            continue
        try:
            df = pl.read_parquet(fp)
        except Exception:
            continue
        for v in SHARED:
            if v not in df.columns:
                continue
            m = analyze(df[v].to_numpy())
            if m:
                rows.append({"grp": r["_grp"], "well_id": r["well_id"],
                             "cls": r["event_class_dir"], "file": fp.name,
                             "variable": v, **m})
        if i % 25 == 0 or i == len(targets):
            print(f"  {i}/{len(targets)}")

    res = pl.DataFrame(rows, infer_schema_length=None)
    res.write_parquet("interpolation_test.parquet")
    print(f"\nSalvo: interpolation_test.parquet ({res.height} series)")

    # ---- (1) por dominio ----
    sec("(1) FRACAO DO SINAL QUE E INTERPOLACAO")
    print("strict   = d2 exatamente zero (limite inferior)")
    print("tolerant = |d2| dentro do quantum de arredondamento")
    print("frozen   = d1 == 0 (congelamento, subconjunto de interpolacao)\n")
    print(f"{'grupo':<14}{'series':>8}{'strict':>10}{'tolerant':>10}"
          f"{'frozen':>9}{'interv.efetivo(s)':>20}")
    print("-" * 71)
    for g in ("real_c9", "real_outros", "simulado"):
        s = res.filter(pl.col("grp") == g)
        if not s.height:
            continue
        eff = s["eff_interval_s"].drop_nulls().drop_nans()
        print(f"{g:<14}{s.height:>8}"
              f"{s['strict_interp'].mean():>10.3f}"
              f"{s['tolerant_interp'].mean():>10.3f}"
              f"{s['frozen'].mean():>9.3f}"
              f"{(eff.median() if eff.len() else float('nan')):>20.1f}")

    # ---- (2) concordancia entre os dois metodos ----
    sec("(2) CONCORDANCIA: ESPACAMENTO vs PERIODICIDADE DE |d2|")
    s = res.filter(pl.col("acf_lag") > 0)
    if s.height:
        print(f"{'variavel':<14}{'interv.mediano':>16}{'lag ACF mediano':>18}"
              f"{'pico ACF medio':>17}")
        print("-" * 65)
        for v in SHARED:
            sv = s.filter(pl.col("variable") == v)
            if not sv.height:
                continue
            eff = sv["eff_interval_s"].drop_nulls().drop_nans()
            print(f"{v:<14}"
                  f"{(eff.median() if eff.len() else float('nan')):>16.1f}"
                  f"{sv['acf_lag'].median():>18.1f}"
                  f"{sv['acf_peak'].mean():>17.3f}")
        print("\nSe as duas colunas de intervalo concordarem, a interpolacao")
        print("esta confirmada por duas vias independentes.")

    # ---- (3) por poco: a interpolacao identifica o poco? ----
    sec("(3) INTERVALO EFETIVO POR POCO (instancias reais)")
    rr = res.filter(pl.col("well_id").is_not_null())
    wells = sorted(w for w in rr["well_id"].unique().to_list() if w)
    print(f"{'poco':<14}" + "".join(f"{v[:9]:>11}" for v in SHARED))
    print("-" * (14 + 11 * len(SHARED)))
    sigs = defaultdict(list)
    for w in wells:
        cells, sig = "", []
        for v in SHARED:
            sv = rr.filter((pl.col("well_id") == w) & (pl.col("variable") == v))
            e = sv["eff_interval_s"].drop_nulls().drop_nans()
            val = float(e.median()) if e.len() else float("nan")
            sig.append(round(val) if np.isfinite(val) else -1)
            cells += f"{val:>11.1f}" if np.isfinite(val) else f"{'-':>11}"
        sigs[tuple(sig)].append(w)
        print(f"{w:<14}{cells}")
    print(f"\nAssinaturas de intervalo distintas: {len(sigs)} para {len(wells)} pocos")
    print("Valores em segundos. 1.0 = medicao nativa a 1 Hz.")

    # ---- (4) impacto no protocolo de janelas ----
    sec("(4) IMPACTO NO PROTOCOLO DA CLASSE 9")
    c = res.filter((pl.col("grp") == "real_c9")
                   & pl.col("variable").is_in(["P-MON-CKP", "T-JUS-CKP"]))
    if c.height:
        e = c["eff_interval_s"].drop_nulls().drop_nans()
        med = float(e.median()) if e.len() else float("nan")
        print(f"  intervalo efetivo mediano ........ {med:.1f} s")
        print(f"  medicoes reais por minuto ........ {60/med:.1f}")
        print(f"  medicoes reais por janela (60min)  {3600/med:.0f}")
        if med > 60:
            print("\n  ALERTA: intervalo efetivo maior que a reamostragem de 1 min.")
            print("  A media por minuto esta agregando pontos interpolados, e")
            print("  a janela de 60 min contem menos medicoes independentes")
            print("  do que os 60 pontos assumidos.")
        elif med > 1:
            print("\n  A reamostragem para 1 min continua valida: o intervalo")
            print("  efetivo e menor que o passo de reamostragem.")

    # ---- (5) a quantizacao sobrevive? ----
    sec("(5) PASSO DE QUANTIZACAO: TODAS AS AMOSTRAS vs PONTOS DE MEDICAO")
    print("Se q_breakpoints >> q_all, o passo estimado antes refletia")
    print("granularidade de interpolacao, nao resolucao de sensor.\n")
    print(f"{'variavel':<14}{'q_all':>14}{'q_breakpoints':>16}{'razao':>10}")
    print("-" * 54)
    for v in SHARED:
        sv = res.filter((pl.col("variable") == v)
                        & (pl.col("grp") != "simulado"))
        a = sv["q_all"].drop_nulls().drop_nans()
        b = sv["q_breakpoints"].drop_nulls().drop_nans()
        if not a.len() or not b.len():
            continue
        ma, mb = float(a.median()), float(b.median())
        print(f"{v:<14}{ma:>14.5g}{mb:>16.5g}"
              f"{(mb/ma if ma else float('nan')):>10.2f}")
    print()


if __name__ == "__main__":
    main()