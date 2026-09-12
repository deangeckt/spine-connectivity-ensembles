# Dendritic spines implement specific connectivity and support neuronal ensembles in cortical circuits

[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.06.07.730704-blue.svg)](https://www.biorxiv.org/content/10.64898/2026.06.07.730704v2) [![Code DOI](https://img.shields.io/badge/code-10.5281%2Fzenodo.21104860-blue.svg)](https://doi.org/10.5281/zenodo.21104860) [![Data DOI](https://img.shields.io/badge/data-10.5281%2Fzenodo.21104425-blue.svg)](https://doi.org/10.5281/zenodo.21104425)

This repository provides the official codebase, data analysis pipelines, and supplementary
notes for the preprint **[Dendritic spines implement specific connectivity and support
neuronal ensembles in cortical
circuits](https://www.biorxiv.org/content/10.64898/2026.06.07.730704v2)** by Dean Geckt,
Netanel Ofer, Michael W. Reimann, Rafael Yuste, and Idan Segev.

![alt text](1.png)

---

## What is here

```
download_data_zenodo.py   route 1 to the structural data — download it, minutes
download_data_cave.py     route 2 to the structural data — rebuild it from MICrONS
figures/                  one notebook per figure; each writes its own PDF/PNG beside itself
figures/supplementary/    the same, for figures S1-S15
images/                   hand-made dendrite renders and EM cut-outs for figures 1, 2, S1
scripts/                  the activity pipeline (figures 6, 7, S14, S15)
utils/                    the analysis code every notebook imports
data/                     created by the two download scripts; not in git
```


| | figures | what it needs |
|---|---|---|
| **structural** | 1-5, S1-S13 | `data/` from either route below |
| **activity** | 6, 7, S14, S15 | the above, **plus** the MICrONS calcium recordings, which you build yourself |

Everything is pinned to MICrONS materialization version **1718**, set by
`MATERIALIZATION_VERSION` in `utils/connectome_types.py`. Root ids change between
materializations, so both routes below use this one.

## Install

```
pip install -r requirements.txt
```

Python 3.12 is what this was developed and tested on.

---

## Step 1 — the structural data

Two routes, same result. Pick one.

### Route 1: download it (minutes)

```
python download_data_zenodo.py
```

Pulls `data.zip` from Zenodo and unpacks it into `data/` — 2.0 GB down, 4.9 GB on disk,
no account needed. Safe to re-run. `--delete-zip` drops the archive afterwards;
`--force` overwrites an existing `data/`.


### Route 2: rebuild it from CAVE (hours)

```
python download_data_cave.py                          # everything, in order
python download_data_cave.py --steps raw,column,meshes # only these
```

Use this to reproduce how we pull and format the raw CAVE data. It needs a CAVE account —
the [MICrONS tutorial](https://tutorial.microns-explorer.org) shows how to set one up.

If a run is interrupted, run the same command again and it picks up where it stopped.
`--steps` runs a subset:

| step | writes | size |
|---|---|---|
| `raw` | `data/raw_tables/` — the seven CAVE tables | 65 MB |
| `column` | `data/micro_column_network/` — the 1,351-cell column | ~1.9 GB |
| `subnets` | `data/micro_column_network/subnetworks/` — figure 5, derived offline | ~780 MB |
| `meshes` | `data/micro_column_network/meshes/` — needed by figure 1, optional for 2, 7, S1 | ~1.2 GB |
| `axon_pr` | `data/all_axon_pr_network/` — figure S3 only | ~960 MB |

`raw,column,subnets,meshes` is the subset worth running: every structural figure except
S3. Keep `meshes` in — figure 1 always draws its cells from them. `axon_pr` buys figure
S3 and nothing else.

### Where the two routes differ

Both routes leave you with the same `data/`. The CAVE route also leaves per-cell
intermediates behind — the `neurons_spines_*` folders under each network, a resume cache
it combines into the spine tables. They are not in the Zenodo archive, and you can delete
them once the run is done.

One difference in the contents, and it changes no figure. In the Zenodo archive, cell
`864691136620192653` has `-1` for `axon_length`, `dendrite_length` and two dendrite
distance columns — its skeleton had not been fetched when that table was built. The
skeleton itself is fine and ships in both routes. A rebuild fills those columns in, so
the cell passes `filter_valid_neuron_w_spines` and the filtered count is 1,140 rather
than 1,139. `download_data_cave.py`'s docstring has two smaller ones.

The dendrite renders and EM cut-outs behind figures 1, 2 and S1 come with neither route:
they were made by hand, so they ship in `images/` in this repository.

---

## Step 2 — the activity half

Figures 6, 7, S14 and S15 also read the MICrONS *functional* recordings: two HDF5 files
under `data/activity/`.

| file | size | what it holds |
|---|---|---|
| `coreg_manual_v4_calcium_v2.h5` | ~19 GB | per-unit calcium and deconvolved spike traces, keyed by `nucleus_id` |
| `microns_per_scan_stimuli.h5` | ~68 MB | per-scan trial table (the oracle clip windows) and condition metadata |

**Neither file is downloadable from here.** At 19 GB the calcium file is far too large to
redistribute, so it is in neither Zenodo nor CAVE — it comes out of the MICrONS
`microns_phase3_nda` DataJoint database, which is only reachable from inside that
project's own docker container.

### 2a. Extract the two H5 files (~1 hour)

Open **`scripts/extract_calcium_data_via_docker.ipynb`** and follow its first cell. It
carries the step-by-step; in outline:

1. Bring up the [MICrONS database container](https://github.com/cajal/microns-nda-access?tab=readme-ov-file#database-container).
2. Copy that notebook into it, together with `data/raw_tables/coregistration_manual_v4.csv`
   — step 1 put it there, whichever route you took.
3. Run the notebook top to bottom. It probes one unit, then does a 20-nucleus dry run,
   then the full pass — hours, and ~19 GB written.
4. Copy the two `.h5` files back into `data/activity/` here.

### 2b. Run the ensemble pipeline (~1 hour)

```
python scripts/ensemble_run.py
```

Detects the ensembles and runs the matched-control bootstraps behind figures 6, S14 and
S15. It writes into `data/activity/ensembles/`, per
detector (`lds`, the ICA detector of the main text, and `ecker`, the population-event
detector used as an independent check):

```
data/activity/ensembles/lds_oracle-resid_disjoint_distance_deg_allmem.pkl
data/activity/ensembles/ecker_oracle-resid_disjoint_distance_deg_allmem.pkl
```

Run it once. No figure notebook runs the pipeline itself — they only read these pickles.

Figure 7 and figure 6 panel A do not use the pickles: they read the calcium H5 directly,
so step 2a alone is enough for them.

---

## Reproducing the figures

One notebook per figure. Open it and run all cells; each writes its output beside itself.

```
figures/figure1.ipynb              ->  figures/fig1_300.png
figures/figure2.ipynb              ->  figures/fig_2.png
figures/figure3.ipynb              ->  figures/fig3.pdf
figures/figure4.ipynb              ->  figures/fig4.pdf
figures/figure5.ipynb              ->  figures/fig5.pdf
figures/figure6.ipynb              ->  figures/fig6.pdf          (needs step 2)
figures/figure7.ipynb              ->  figures/fig7.pdf          (needs step 2a)
figures/supplementary/fig_s1.ipynb ->  figures/supplementary/fig_s1.png
...                                    (S14, S15 need step 2)
```

### Meshes or skeletons?

Figures 1, 2, 7 and S1 draw reconstructed cells, and a cell can be drawn either from its
segmented surface mesh or from its skeleton. The meshes are what make the beautiful cells
in the paper. They live in `data/micro_column_network/meshes/` — ~1.2 GB, all ten cells
by either route — and are markedly slower to draw.

**Figure 1 always draws from the meshes**, so it needs that folder.

For figures 2, 7 and S1 **skeletons are the default**: all 1,351 of them ship in `data/`,
they load in seconds, and every panel reads correctly from them. To draw those cells from
meshes as well, flip the flag near the top of the notebook:

```
USE_MESHES = True        # figures 2, S1
CELLS_MODE = 'meshes'    # figure 7
```

### The pre-rendered dendrite panels

The zoomed dendrite insets in figures 1, 2 and S1 — and the EM cut-outs beside them in
figure 1 — are a separate thing. They are not drawn from `data/`: they are images, and
they ship in this repository, 12 MB in four folders:

```
images/fig1_branches/       the excitatory and inhibitory dendrites of figure 1
images/fig1_syn_em_images/  the four EM cut-outs beside them
images/fig2_branches/       the four spiny branches of figure 2
images/fig2_inh_branches/   the four aspiny branches of figure S1
```

The branch renders were segmented and rendered by hand, one panel at a time, in
[VAST](https://lichtman.rc.fas.harvard.edu/vast/). Nothing in CAVE serves them, so
**neither route in step 1 can rebuild them** — which is why they sit in git rather than in
`data/`. Clone the repository and they are there, whichever route you took.

The four EM cut-outs are the one part that is reproducible: `figure1.ipynb` fetches them
through imageryclient when `images/fig1_syn_em_images/` is empty, saves them as `.npy`,
and reads the saved arrays on every run after that.

The notebooks read these images whenever the folder holds them — the zoom boxes and
synapse markers in those panels are positioned against these exact renders, so the insets
look right either way. Emptying the matching `images/` folder is what makes a notebook
draw an inset live from the mesh instead (in figures 2 and S1, with `USE_MESHES = True`
as well).

Notebooks add `utils/` to the path in their first cell and resolve `data/` relative to the
repo, so run them from where they sit and they will find everything. A notebook that needs
activity data checks for it in its first cells and fails with build instructions rather
than an unreadable error deep in the run.

---

## Conventions

### Connectivity matrix
`filtered_syn_mat[i, j]` = synapses from neuron `i` (pre) → neuron `j` (post).
`filtered_mapping` = `{matrix_index: root_id}`. Always rebuild after any neuron filtering.


---

## Citation

If you use this code or data in your research, please cite our preprint:

```bibtex
@misc{geckt2026dendriticspines,
	title = {Dendritic spines implement specific connectivity and support neuronal ensembles in cortical circuits},
	author = {Geckt, Dean and Ofer, Netanel and Reimann, Michael W. and Yuste, Rafael and Segev, Idan},
	year = {2026},
	doi = {10.64898/2026.06.07.730704},
	url = {https://www.biorxiv.org/content/10.64898/2026.06.07.730704v2},
	note = {Preprint, version 2}
}
```

---

## Operating system

Everything here was developed and run on **Windows 11** with Python 3.12. Nothing in the
code is platform-specific — paths are built with `os.path`, and every pinned dependency
publishes Linux and macOS wheels — so Linux and macOS should work too, but we have not
tested them.
