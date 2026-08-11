#!/usr/bin/env python3
"""
step11_sampling_interval.py
===========================
Corrige o estimador de intervalo efetivo do passo 10.

DEFEITOS CORRIGIDOS
  (1) Pareamento de quebras. d2 usa tres amostras consecutivas, entao uma
      unica medicao real quebra DOIS valores de d2. As quebras aparecem
      em pares e a moda dos espacamentos passa a ser 1, medindo a distancia
      interna ao par em vez do intervalo entre medicoes.
      -> quebras consecutivas sao agrupadas em um unico evento de medicao.

  (2) Tolerancia absoluta. tol = menor variacao positiva funciona para
      variaveis quantizadas (pressoes), mas para as continuas essa
      quantidade e ruido de ponto flutuante (1e-5), a tolerancia vira zero
      e tudo e classificado como quebra.
      -> criterio relativo, calibrado pela dispersao de d2.

  (3) Harmonicos na ACF. O argmax pode cair em 2k ou 3k.
      -> reporta o histograma completo de espacamentos, nao so um resumo,
         e verifica se o lag da ACF e multiplo do espacamento modal.

Uso:
    python step11_sampling_interval.py <raiz> [manifesto] [--sample=40]
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

SHARED = ["P-MON-CKP", "P-PDG", "P-TPT", "T-JUS-CKP", "T-TPT"]
MIN_LEN = 3000
MERGE_GAP = 2        # quebras separadas por <= isto pertencem ao mesmo evento
REL_TOL = 0.05       # |d2| <= REL_TOL * MAD(d2) => interpolacao
MAX_LAG = 600
LINE = "=" * 78


def sec(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


def longest_valid(x):
    ok = np.isfinite(x)
    if not ok.any():
        return None
    d = np.diff(np.r_[0, ok.view(np.int8), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    i = np.argmax(e - s)
    return x[s[i]:e[i]]


def acf_peak(y, max_lag):
    y = y - y.mean()
    n = len(y)
    if n < 4 * max_lag or np.allclose(y, 0):
        return None
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[:max_lag + 1]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    return int(np.argmax(ac[2:max_lag + 1])) + 2


def analyze(x):
    seg = longest_valid(np.asarray(x, dtype=np.float64))
    if seg is None or seg.size < MIN_LEN:
        return None
    d1 = np.diff(seg)
    d2 = np.diff(d1)
    if d2.size < 200:
        return None

    # (2) tolerancia relativa, robusta a escala e a natureza da variavel
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    pos = np.abs(d1[d1 != 0])
    q_abs = float(pos.min()) if pos.size else 0.0
    tol = max(REL_TOL * mad, q_abs * 1e-9)

    interp = np.abs(d2) <= tol
    frac_interp = float(np.mean(interp))

    brk = np.flatnonzero(~interp)
    if brk.size < 20:
        return {"frac_interp": frac_interp, "n_events": 0,
                "spacing_mode": -1, "spacing_median": float("nan"),
                "acf_lag": -1, "harmonic_ok": None, "q_meas": float("nan")}

    # (1) agrupa quebras consecutivas em eventos de medicao
    split = np.flatnonzero(np.diff(brk) > MERGE_GAP)
    groups = np.split(brk, split + 1)
    centers = np.array([g.mean() for g in groups])
    spacing = np.diff(centers)
    spacing = spacing[spacing > 0]
    if spacing.size < 10:
        return {"frac_interp": frac_interp, "n_events": len(groups),
                "spacing_mode": -1, "spacing_median": float("nan"),
                "acf_lag": -1, "harmonic_ok": None, "q_meas": float("nan")}

    rounded = np.round(spacing).astype(int)
    mode = Counter(rounded.tolist()).most_common(1)[0][0]
    med = float(np.median(spacing))

    lag = acf_peak(np.abs(d2), min(MAX_LAG, d2.size // 4))
    harmonic = None
    if lag and mode > 0:
        ratio = lag / mode
        harmonic = bool(abs(ratio - round(ratio)) < 0.15 and 1 <= round(ratio) <= 4)

    # quantizacao nos valores efetivamente medidos
    vals = seg[np.clip(np.round(centers).astype(int), 0, seg.size - 1)]
    dv = np.abs(np.diff(vals))
    dv = dv[dv > 0]
    q_meas = float(np.min(dv)) if dv.size else float("nan")

    return {"frac_interp": frac_interp, "n_events": len(groups),
            "spacing_mode": int(mode), "spacing_median": med,
            "acf_lag": lag if lag else -1, "harmonic_ok": harmonic,
            "q_meas": q_meas,
            "_hist": Counter(rounded.tolist())}


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
    mf = pl.read_parquet(man)

    c9 = mf.filter((pl.col("event_class_dir") == 9)
                   & (pl.col("instance_type") == "real"))
    others = mf.filter((pl.col("event_class_dir") != 9)
                       & (pl.col("instance_type") == "real"))
    sims = mf.filter(pl.col("instance_type") == "simulated")

    def take(df, n):
        if df.height <= n:
            return df.to_dicts()
        return df.to_dicts()[::max(1, df.height // n)][:n]

    targets = ([{**r, "_g": "real_c9"} for r in c9.to_dicts()]
               + [{**r, "_g": "real_outros"} for r in take(others, sample_n)]
               + [{**r, "_g": "simulado"} for r in take(sims, sample_n)])

    print(f"Analisando {len(targets)} instancias\n")
    rows, hists = [], {v: Counter() for v in SHARED}

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
            if not m:
                continue
            h = m.pop("_hist", None)
            if h is not None and r["_g"] != "simulado":
                hists[v].update(h)
            rows.append({"grp": r["_g"], "well_id": r["well_id"],
                         "variable": v, "file": fp.name, **m})
        if i % 25 == 0 or i == len(targets):
            print(f"  {i}/{len(targets)}")

    res = pl.DataFrame(rows, infer_schema_length=None)
    res.write_parquet("sampling_interval.parquet")
    print(f"\nSalvo: sampling_interval.parquet ({res.height} series)")

    sec("(1) INTERVALO EFETIVO APOS AGRUPAMENTO DE QUEBRAS")
    print(f"{'grupo':<14}{'variavel':<14}{'interp':>9}{'moda(s)':>10}"
          f"{'mediana(s)':>12}{'ACF':>7}{'harmonico?':>12}")
    print("-" * 78)
    for g in ("real_c9", "real_outros", "simulado"):
        for v in SHARED:
            s = res.filter((pl.col("grp") == g) & (pl.col("variable") == v)
                           & (pl.col("spacing_mode") > 0))
            if not s.height:
                continue
            hm = s.filter(pl.col("harmonic_ok").is_not_null())
            frac_h = (hm["harmonic_ok"].mean() if hm.height else float("nan"))
            print(f"{g:<14}{v:<14}{s['frac_interp'].mean():>9.3f}"
                  f"{s['spacing_mode'].median():>10.1f}"
                  f"{s['spacing_median'].median():>12.1f}"
                  f"{s['acf_lag'].median():>7.0f}"
                  f"{frac_h:>12.2f}")
        print()
    print("'harmonico?' = fracao de series em que o lag da ACF e multiplo")
    print("inteiro da moda. Proximo de 1 confirma as duas vias.")

    sec("(2) HISTOGRAMA DE ESPACAMENTOS (instancias reais)")
    print("Um pico unico indica cadencia fixa; varios picos indicam")
    print("cadencias misturadas dentro das proprias series.\n")
    for v in SHARED:
        h = hists[v]
        if not h:
            continue
        tot = sum(h.values())
        top = h.most_common(8)
        print(f"{v}  (n={tot})")
        for k, c in sorted(top):
            bar = "#" * int(60 * c / tot)
            print(f"    {k:>5} s  {100*c/tot:>6.1f}%  {bar}")
        print()

    sec("(3) INTERVALO POR POCO (moda, instancias reais)")
    rr = res.filter(pl.col("well_id").is_not_null()
                    & (pl.col("spacing_mode") > 0))
    wells = sorted(w for w in rr["well_id"].unique().to_list() if w)
    print(f"{'poco':<14}" + "".join(f"{v[:9]:>11}" for v in SHARED))
    print("-" * (14 + 11 * len(SHARED)))
    sigs = set()
    for w in wells:
        cells, sig = "", []
        for v in SHARED:
            s = rr.filter((pl.col("well_id") == w) & (pl.col("variable") == v))
            val = float(s["spacing_mode"].median()) if s.height else float("nan")
            sig.append(round(val) if np.isfinite(val) else -1)
            cells += f"{val:>11.0f}" if np.isfinite(val) else f"{'-':>11}"
        sigs.add(tuple(sig))
        print(f"{w:<14}{cells}")
    print(f"\nAssinaturas distintas: {len(sigs)} para {len(wells)} pocos")
    print()


if __name__ == "__main__":
    main()