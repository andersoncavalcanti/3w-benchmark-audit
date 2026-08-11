# 3w-audit

Replication code for a structural audit of the [3W dataset](https://doi.org/10.6084/m9.figshare.29205836.v1),
a public collection of real and simulated time series from offshore oil wells.

This repository accompanies the paper:

> Cavalcanti, A. L. O., Maitelli, C. W. S. P., Brito, H. G. (2026).
> *Structural confounds in the 3W dataset: instrumental disjointness,
> class–well entanglement, and their effect on reported detection performance.*
> <!-- TODO: journal, volume, DOI once accepted -->

## What this code establishes

| Finding | Script | Figure |
|---|---|---|
| Real and simulated subsets share only 5 of 27 variables, and no availability signature | 1, 2 | Fig. 2 |
| Missingness is tag-level, not sample-level (98.1% of pairs fully present or fully absent) | 3 | Fig. 3 |
| 24.9% of non-absent real signal is invariant within its instance | 4 | Fig. 4 |
| 91% of the real signal is piecewise-linear reconstruction; 26% in the simulated control | 10 | Fig. 7 |
| Event class is confounded with well identity | 2 | Fig. 1 |
| Ungrouped validation inflates AUROC by 0.385 and inverts model ranking | 8, 9 | Figs. 5, 6 |

## Requirements

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

The full pipeline runs on a laptop CPU. The heaviest step (4) reads all 2228
Parquet files and takes a few minutes.

## Getting the data

The dataset is **not** included here. Download 3W Dataset 2.0.0 from
[Figshare](https://doi.org/10.6084/m9.figshare.29205836.v1) and extract it so
that the directory tree looks like:

```
<DATA_ROOT>/
  dataset.ini
  0/  1/  2/ ... 9/     one directory per event class, Parquet files inside
```

Confirm the version in `dataset.ini` before proceeding. Results reported in
the paper are specific to release 2.0.0; earlier releases differ in the number
of wells, in class numbering, and in which variables exist.

Keep `<DATA_ROOT>` read-only and outside any synchronised folder. Cloud
sync clients that dehydrate files on demand will make step 4 extremely slow.

## Reproducing the results

Run in order from the repository root. Replace `<DATA_ROOT>` with your path.

```bash
# 1. Instance-level manifest: one row per instance, 149 columns
python code/build_3w_manifest.py <DATA_ROOT>

# 2. Availability signatures, well x class matrix, domain separability
python code/analyze_3w_manifest.py manifest_3w.parquet

# 3. Missingness histogram and class-9 label budget
python code/step3_quality_forensics.py <DATA_ROOT> manifest_3w.parquet

# 4. Signal quality taxonomy: absent / constant / frozen / quantised / continuous
python code/step4_signal_taxonomy.py <DATA_ROOT> manifest_3w.parquet

# 5. Which class-9 instances are actually usable
python code/step5_class9_viability.py <DATA_ROOT> manifest_3w.parquet taxonomy_pairs.parquet

# 6. Choice of variable subset, evaluated over all 31 non-empty subsets
python code/step6_feature_subset.py class9_viability.parquet manifest_3w.parquet

# 7. Window index for the detection task
python code/step7_build_windows.py <DATA_ROOT> class9_viability.parquet manifest_3w.parquet

# 8. The leakage budget: three protocols, three normalisations, ten seeds
python code/step8_experiment.py windows_class9.npz --seeds=10

# 9. Well identifiability and seed variance
python code/step9_shortcut_and_variance.py windows_class9.npz --seeds=10

# 10. Interpolation test against the simulated negative control
python code/step10_interpolation_test.py <DATA_ROOT> manifest_3w.parquet

# Figures
python code/make_figures.py
```

Steps 1–6 are characterisation and produce the numbers in Sections 5 and 7.1
of the paper. Steps 7–9 produce the leakage budget in Section 6. Step 10
produces Section 5.4.

### Derived artefacts

The `outputs/` directory contains the artefacts produced by steps 1, 4, 5, 8,
9 and 10. They are small, and including them lets you regenerate every figure
without downloading the dataset:

```bash
python code/make_figures.py
```

## Notes on two analyses

**Effective sampling interval.** Step 10 establishes robustly that most of the
real signal is reconstruction rather than measurement, using the simulated
subset as a negative control. It does *not* establish the effective
measurement interval. Two estimators were attempted and both failed: spacing
between measurement instants is biased downward because one measurement breaks
two consecutive second differences, and the periodicity of the
second-difference magnitude is vulnerable to harmonics. A third attempt using
a relative tolerance failed the negative control outright. The script
`step11_sampling_interval.py` is included for completeness and its output
should **not** be used; the paper reports the fraction only and leaves the
interval open.

**Well identifiability.** Step 9 currently reports unbalanced accuracy. One
well contributes 37% of the windows, so the majority-class baseline is
dominated by it. Balanced accuracy is the appropriate metric; see the TODO in
the script.

## Repository layout

```
code/       analysis scripts, numbered in execution order
outputs/    derived artefacts (Parquet, npz) — small, version-controlled
figs/       figures as vector PDF
```

## Citing

If you use this code, please cite the paper above and the dataset:

> Vargas, R. E. V. et al. (2019). A realistic and public dataset with rare
> undesirable real events in oil wells. *Journal of Petroleum Science and
> Engineering*, 181, 106223.

<!-- TODO: add the Zenodo DOI badge once a release is archived -->

## License

MIT. See [LICENSE](LICENSE).

The 3W dataset itself is distributed separately by its maintainers under
CC BY 4.0 and is not redistributed here.
