#!/usr/bin/env python3
"""
step6_feature_subset.py
=======================
Escolhe o subconjunto FIXO de variaveis para a demonstracao empirica.

O criterio "pelo menos k de 5 uteis" e diagnostico, nao um protocolo: um
modelo precisa de um espaco de features fixo. Este script avalia todos os
31 subconjuntos nao vazios de {P-MON-CKP, P-PDG, P-TPT, T-JUS-CKP, T-TPT},
exigindo que TODAS as variaveis do subconjunto sejam uteis (QUANTIZADA ou
CONTINUA) na instancia.

Para cada subconjunto reporta:
  - instancias elegiveis, positivas (contem rotulo 109) e negativas
  - pocos, e pocos com pelo menos uma positiva  (= numero de dobras LOWO)
  - EVENTOS distintos: instancias positivas do mesmo poco separadas por
    menos de GAP_DAYS sao contadas como um unico evento
  - horas de sinal disponiveis

Uso:
    python step6_feature_subset.py [viabilidade] [manifesto] [--gap-days=7]
"""

import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import polars as pl

SHARED = ["P-MON-CKP", "P-PDG", "P-TPT", "T-JUS-CKP", "T-TPT"]
USEFUL = {"QUANTIZADA", "CONTINUA"}
MIN_FOLDS = 5
LINE = "=" * 78


def parse_ts(filename: str):
    """WELL-00042_20141218004109.parquet -> datetime"""
    try:
        stem = Path(filename).stem
        return datetime.strptime(stem.split("_")[1], "%Y%m%d%H%M%S")
    except Exception:
        return None


def count_events(rows, gap_days):
    """Agrupa positivas do mesmo poco separadas por < gap_days."""
    by_well = {}
    for r in rows:
        ts = parse_ts(r["filename"])
        if ts is None:
            continue
        by_well.setdefault(r["well_id"], []).append(ts)
    total = 0
    for w, ts_list in by_well.items():
        ts_list.sort()
        n = 1
        for a, b in zip(ts_list, ts_list[1:]):
            if (b - a).days >= gap_days:
                n += 1
        total += n
    return total


def main():
    pos_args, gap_days = [], 7
    for a in sys.argv[1:]:
        if a.startswith("--gap-days"):
            gap_days = int(a.split("=", 1)[1])
        elif a.startswith("--"):
            print(f"Opcao desconhecida: {a}")
            sys.exit(1)
        else:
            pos_args.append(a)

    via = Path(pos_args[0]) if pos_args else Path("class9_viability.parquet")
    man = Path(pos_args[1]) if len(pos_args) > 1 else Path("manifest_3w.parquet")
    for p in (via, man):
        if not p.exists():
            print(f"ERRO: nao encontrado: {p}")
            sys.exit(1)

    df = pl.read_parquet(via)
    mf = pl.read_parquet(man)

    # duracao por arquivo
    dur = {Path(r["filepath"]).name: (r["duration_s"] or 0) / 3600
           for r in mf.select(["filepath", "duration_s"]).to_dicts()}

    rows = df.to_dicts()
    for r in rows:
        r["hours"] = dur.get(r["filename"], 0.0)

    print(f"Instancias reais classe 9: {len(rows)}")
    print(f"Agrupamento de eventos: instancias do mesmo poco separadas por "
          f"< {gap_days} dias contam como um evento.\n")

    results = []
    for k in range(1, len(SHARED) + 1):
        for combo in combinations(SHARED, k):
            elig = [r for r in rows
                    if all(r[f"cat_{v}"] in USEFUL for v in combo)]
            if not elig:
                continue
            posr = [r for r in elig if r["positive"]]
            negr = [r for r in elig if not r["positive"]]
            wells_pos = {r["well_id"] for r in posr}
            results.append({
                "combo": combo, "k": k,
                "n": len(elig), "pos": len(posr), "neg": len(negr),
                "wells": len({r["well_id"] for r in elig}),
                "folds": len(wells_pos),
                "events": count_events(posr, gap_days),
                "h_pos": sum(r["hours"] for r in posr),
                "h_neg": sum(r["hours"] for r in negr),
            })

    # ordena por dobras, depois eventos, depois numero de variaveis
    results.sort(key=lambda r: (-r["folds"], -r["events"], -r["k"], -r["n"]))

    print(LINE)
    print("TODOS OS SUBCONJUNTOS (ordenados por dobras LOWO, depois eventos)")
    print(LINE)
    hdr = (f"{'variaveis':<44}{'inst':>6}{'pos':>5}{'neg':>5}"
           f"{'poços':>7}{'dobras':>8}{'event':>7}{'h_pos':>8}{'h_neg':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        name = "+".join(v.replace("-", "") for v in r["combo"])
        flag = "  <<" if r["folds"] >= MIN_FOLDS and r["k"] >= 3 else ""
        print(f"{name:<44}{r['n']:>6}{r['pos']:>5}{r['neg']:>5}"
              f"{r['wells']:>7}{r['folds']:>8}{r['events']:>7}"
              f"{r['h_pos']:>8.1f}{r['h_neg']:>8.1f}{flag}")

    print(f"\n<< = pelo menos {MIN_FOLDS} dobras LOWO e ao menos 3 variaveis")

    # Recomendacao
    print(f"\n{LINE}\nRECOMENDACAO\n{LINE}")
    viable = [r for r in results if r["folds"] >= MIN_FOLDS and r["k"] >= 3]
    if not viable:
        print("Nenhum subconjunto com >=3 variaveis atinge o minimo de dobras.")
        print("Opcoes: reduzir para 2 variaveis, aceitar menos dobras, ou")
        print("substituir a demonstracao por reanalise de metodo publicado.")
        alt = [r for r in results if r["folds"] >= MIN_FOLDS]
        if alt:
            b = alt[0]
            print(f"\nMelhor com >= {MIN_FOLDS} dobras: "
                  f"{'+'.join(b['combo'])} ({b['k']} variaveis, "
                  f"{b['folds']} dobras, {b['events']} eventos)")
    else:
        best = max(viable, key=lambda r: (r["events"], r["k"], r["n"]))
        print(f"  variaveis .......... {', '.join(best['combo'])}")
        print(f"  instancias ......... {best['n']} "
              f"({best['pos']} positivas / {best['neg']} negativas)")
        print(f"  dobras LOWO ........ {best['folds']}")
        print(f"  eventos distintos .. {best['events']}")
        print(f"  horas de sinal ..... {best['h_pos']:.1f} positivas / "
              f"{best['h_neg']:.1f} negativas")
        print("\n  Detalhe por dobra:")
        elig = [r for r in rows
                if all(r[f"cat_{v}"] in USEFUL for v in best["combo"])]
        wells = sorted({r["well_id"] for r in elig})
        for w in wells:
            sw = [r for r in elig if r["well_id"] == w]
            p = [r for r in sw if r["positive"]]
            print(f"    {w}: {len(sw):>2} instancias "
                  f"({len(p)} pos, {len(sw)-len(p)} neg), "
                  f"{sum(r['hours'] for r in sw):>6.1f} h, "
                  f"{count_events(p, gap_days)} evento(s)")
        print("\n  ATENCAO: verifique se uma unica dobra concentra a maioria")
        print("  das positivas. Se sim, reporte metricas POR DOBRA, nunca so")
        print("  a media, e declare o numero de eventos junto com o de")
        print("  instancias.")
    print()


if __name__ == "__main__":
    main()