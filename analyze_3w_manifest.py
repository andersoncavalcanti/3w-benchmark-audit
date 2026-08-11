#!/usr/bin/env python3
"""
analyze_3w_manifest.py
======================
Passo 2 da Linha A. Roda 100% sobre manifest_3w.parquet (nao rele os dados).

Responde tres perguntas que decidem o desenho experimental:

  A) COBERTURA REAL das 5 variaveis compartilhadas em nivel de observacao.
     ('present' apenas diz "tem >=1 valor nao-nulo" -- e um limite superior.)

  B) MATRIZ POCO x CLASSE. Define se o protocolo leave-one-well-out e
     possivel, e de onde vem a classe negativa (normalidade).

  C) TESTE DE ATALHO. Quantas assinaturas de presenca de variaveis sao
     compartilhadas entre dominio real e simulado? Se ~zero, a assinatura
     de faltantes identifica o dominio -- e, por consequencia, vaza a
     classe em qualquer treino que misture real e simulado.

Uso:
    python analyze_3w_manifest.py manifest_3w.parquet
"""

import sys
from collections import defaultdict

import polars as pl

SHARED = ["P-MON-CKP", "P-PDG", "P-TPT", "T-JUS-CKP", "T-TPT"]

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

# Limiares de usabilidade de uma instancia (fracao maxima de faltantes)
THRESHOLDS = [0.01, 0.05, 0.20, 0.50]

TARGET_CLASSES = [2, 8, 9]   # classes que passaram nos dois gates
LINE = "=" * 78


def sec(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


# ---------------------------------------------------------------------------
# A) Cobertura em nivel de observacao
# ---------------------------------------------------------------------------

def coverage(mf):
    sec("A) COBERTURA OBSERVACIONAL DAS 5 VARIAVEIS COMPARTILHADAS")
    print("Fracao MEDIA de observacoes ausentes, por classe e dominio.")
    print("0.00 = variavel integralmente disponivel; 1.00 = inexistente.\n")

    for dom in ("real", "simulated"):
        print(f"--- dominio: {dom} ---")
        hdr = f"{'classe':>7}{'n':>6}" + "".join(f"{v:>13}" for v in SHARED)
        print(hdr)
        print("-" * len(hdr))
        for c in range(10):
            sub = mf.filter(
                (pl.col("event_class_dir") == c)
                & (pl.col("instance_type") == dom)
            )
            if not sub.height:
                continue
            vals = "".join(
                f"{float(sub[f'{v}__missing_frac'].mean()):>13.3f}"
                for v in SHARED
            )
            print(f"{c:>7}{sub.height:>6}{vals}")
        print()

    sec("A2) INSTANCIAS COM AS 5 VARIAVEIS SIMULTANEAMENTE USAVEIS")
    print("Uma instancia so serve ao estudo sim->real se TODAS as 5")
    print("variaveis estiverem abaixo do limiar de faltantes.\n")

    for thr in THRESHOLDS:
        cond = pl.lit(True)
        for v in SHARED:
            cond = cond & (pl.col(f"{v}__missing_frac") <= thr)
        ok = mf.filter(cond)
        print(f"  limiar <= {thr:.2f} de faltantes:")
        for c in TARGET_CLASSES + [0]:
            r = ok.filter((pl.col("event_class_dir") == c)
                          & (pl.col("instance_type") == "real")).height
            s = ok.filter((pl.col("event_class_dir") == c)
                          & (pl.col("instance_type") == "simulated")).height
            w = ok.filter((pl.col("event_class_dir") == c)
                          & (pl.col("well_id").is_not_null()))["well_id"].n_unique()
            print(f"     classe {c}: real={r:>4}  simul={s:>4}  pocos={w:>3}")
        print()


# ---------------------------------------------------------------------------
# B) Matriz poco x classe
# ---------------------------------------------------------------------------

def well_matrix(mf):
    sec("B) MATRIZ POCO x CLASSE (apenas instancias reais)")

    real = mf.filter(pl.col("instance_type") == "real")
    tab = defaultdict(lambda: defaultdict(int))
    for w, c in zip(real["well_id"].to_list(),
                    real["event_class_dir"].to_list()):
        if w is not None:
            tab[w][c] += 1

    wells = sorted(tab.keys())
    hdr = f"{'poco':<14}" + "".join(f"{c:>5}" for c in range(10)) + f"{'tot':>7}"
    print(hdr)
    print("-" * len(hdr))
    for w in wells:
        tot = sum(tab[w].values())
        cells = "".join(
            (f"{tab[w][c]:>5}" if tab[w][c] else f"{'.':>5}") for c in range(10)
        )
        print(f"{w:<14}{cells}{tot:>7}")
    print("-" * len(hdr))
    tot_row = "".join(
        f"{sum(tab[w][c] for w in wells):>5}" for c in range(10)
    )
    print(f"{'TOTAL':<14}{tot_row}{sum(sum(v.values()) for v in tab.values()):>7}")
    print(f"\nPocos reais distintos: {len(wells)}")

    sec("B2) SOBREPOSICAO COM A CLASSE NEGATIVA (classe 0)")
    print("Para deteccao binaria com leave-one-well-out, o ideal e que o")
    print("poco tenha instancias da classe alvo E instancias normais.\n")

    w0 = {w for w in wells if tab[w][0] > 0}
    for c in TARGET_CLASSES:
        wc = {w for w in wells if tab[w][c] > 0}
        both = wc & w0
        print(f"  classe {c}: {len(wc):>3} pocos com o evento | "
              f"{len(both):>3} tambem com classe 0 "
              f"({100*len(both)/len(wc) if wc else 0:.0f}%)")
        if wc:
            print(f"     com ambos : {sorted(both) if both else '(nenhum)'}")
            print(f"     so evento : {sorted(wc - w0) if wc - w0 else '(nenhum)'}")
        print()
    print("Se a sobreposicao for baixa, a normalidade tera de ser extraida")
    print("do periodo pre-evento DENTRO das proprias instancias do evento.")


# ---------------------------------------------------------------------------
# C) Teste de atalho: assinatura de presenca separa os dominios?
# ---------------------------------------------------------------------------

def shortcut_test(mf):
    sec("C) TESTE DE ATALHO - A ASSINATURA DE FALTANTES IDENTIFICA O DOMINIO?")

    def signature(row):
        return tuple(int(row[f"{v}__present"]) for v in VARIABLES)

    rows = mf.to_dicts()
    sig_real, sig_sim = defaultdict(int), defaultdict(int)
    for r in rows:
        if r["instance_type"] == "real":
            sig_real[signature(r)] += 1
        elif r["instance_type"] == "simulated":
            sig_sim[signature(r)] += 1

    shared = set(sig_real) & set(sig_sim)
    n_real = sum(sig_real.values())
    n_sim = sum(sig_sim.values())
    amb_real = sum(sig_real[s] for s in shared)
    amb_sim = sum(sig_sim[s] for s in shared)

    print(f"  assinaturas distintas no real ......... {len(sig_real)}")
    print(f"  assinaturas distintas no simulado ..... {len(sig_sim)}")
    print(f"  assinaturas compartilhadas ............ {len(shared)}")
    print(f"  instancias reais ambiguas ............. {amb_real}/{n_real} "
          f"({100*amb_real/n_real if n_real else 0:.2f}%)")
    print(f"  instancias simuladas ambiguas ......... {amb_sim}/{n_sim} "
          f"({100*amb_sim/n_sim if n_sim else 0:.2f}%)")

    acc = 1 - (amb_real + amb_sim) / (n_real + n_sim)
    print(f"\n  >>> Acuracia minima de um classificador de dominio que use")
    print(f"      APENAS a mascara de faltantes: {100*acc:.2f}%")
    if acc > 0.95:
        print("      Separabilidade quase perfeita. Em qualquer treino que")
        print("      misture real e simulado, a mascara de faltantes e um")
        print("      atalho disponivel -- e, como as classes raras sao")
        print("      majoritariamente simuladas, ela vaza o rotulo.")

    sec("C2) MESMO TESTE RESTRITO AS 5 VARIAVEIS COMPARTILHADAS")
    print("Se a separabilidade cair aqui, restringir o espaco de features")
    print("as 5 variaveis e uma mitigacao eficaz do atalho.\n")

    def sig5(row):
        return tuple(int(row[f"{v}__present"]) for v in SHARED)

    s_real, s_sim = defaultdict(int), defaultdict(int)
    for r in rows:
        if r["instance_type"] == "real":
            s_real[sig5(r)] += 1
        elif r["instance_type"] == "simulated":
            s_sim[sig5(r)] += 1
    sh = set(s_real) & set(s_sim)
    ar = sum(s_real[s] for s in sh)
    asim = sum(s_sim[s] for s in sh)
    acc5 = 1 - (ar + asim) / (n_real + n_sim)
    print(f"  assinaturas compartilhadas ............ {len(sh)}")
    print(f"  instancias ambiguas ................... {ar + asim}")
    print(f"  >>> Acuracia minima do classificador ... {100*acc5:.2f}%")


# ---------------------------------------------------------------------------
# D) Detalhe da classe alvo
# ---------------------------------------------------------------------------

def class9_detail(mf):
    sec("D) DETALHE DA CLASSE 9 (hidrato em linha de servico) - INSTANCIAS REAIS")
    sub = (mf.filter((pl.col("event_class_dir") == 9)
                     & (pl.col("instance_type") == "real"))
             .sort("well_id"))
    hdr = (f"{'poco':<13}{'n_obs':>9}{'dur_h':>8}"
           f"{'%c0':>7}{'%c109':>8}{'%c9':>7}"
           + "".join(f"{v[:8]:>10}" for v in SHARED))
    print(hdr)
    print("-" * len(hdr))
    for r in sub.to_dicts():
        dur = (r["duration_s"] or 0) / 3600
        f0 = (r.get("class_0_frac") or 0) * 100
        f109 = (r.get("class_109_frac") or 0) * 100
        f9 = (r.get("class_9_frac") or 0) * 100
        cov = "".join(
            f"{1 - (r[f'{v}__missing_frac'] or 1):>10.2f}" for v in SHARED
        )
        print(f"{str(r['well_id']):<13}{r['n_obs']:>9}{dur:>8.1f}"
              f"{f0:>7.1f}{f109:>8.1f}{f9:>7.1f}{cov}")
    print("\n(as 5 ultimas colunas sao COBERTURA: 1.00 = variavel completa)")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "manifest_3w.parquet"
    mf = pl.read_parquet(path)
    print(f"Manifesto: {mf.height} instancias x {mf.width} colunas")
    coverage(mf)
    well_matrix(mf)
    shortcut_test(mf)
    class9_detail(mf)
    print()


if __name__ == "__main__":
    main()