# Connectivity Logic of Dendritic Spines in Cortex: Increased Inputs and Ensemble Formation

[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.06.07.730704-blue.svg)](https://doi.org/10.64898/2026.06.07.730704) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21104860.svg)](https://doi.org/10.5281/zenodo.21104860)


This repository provides the official codebase, data analysis pipelines, and supplementary notes for the preprint **[Connectivity Logic of Dendritic Spines in Cortex: Increased Inputs and Ensemble Formation](https://doi.org/10.64898/2026.06.07.730704)** by Dean Geckt, Netanel Ofer, Michael W. ReimaNnn, Rafael Yuste, and Idan Segev.


![alt text](1.png)

## Getting started
To reproduce the analyses, first run the `fetch_zenodo_data.py` script to automatically download and extract the required dataset. Once the data directory is populated, you can execute any of the provided Jupyter notebooks to generate the corresponding manuscript figures.

You may need to install a python env via the `requirements.txt`

## Citation

If you use this code or data in your research, please cite our preprint:

```bibtex
@misc{geckt2026connectivitylogic,
	title = {Connectivity Logic of Dendritic Spines in Cortex: Increased Inputs and Ensemble Formation},
	author = {Geckt, Dean and Ofer, Netanel and Reimann, Michael W. and Yuste, Rafael and Segev, Idan},
	year = {2026},
	doi = {10.64898/2026.06.07.730704},
	url = {https://doi.org/10.64898/2026.06.07.730704},
	note = {Preprint}
}
```


## Conventions

### Data version
`MATERIALIZATION_VERSION = 1718` is defined in `connectome_types.py` and must be passed to any CAVEclient call: `client.materialize.version = MATERIALIZATION_VERSION`.


### Null models (figure 4 only)
Generated in-memory with `generate_shuffles(bin_mat, mapping, neuron_clf_type, shuffle_mode, amount)`. No disk cache. `shuffle_mode` is `'cfg'` (configuration model).


### Connectivity matrix convention
`filtered_syn_mat[i, j]` = synapses from neuron `i` (pre) → neuron `j` (post).
`filtered_mapping` = `{matrix_index: root_id}`. Always rebuild after any neuron filtering.
