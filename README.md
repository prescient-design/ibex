# Ibex 🐐

Ibex is a lightweight antibody and TCR structure prediction model.

<p align="center">
<img src="docs/assets/ibex.png" width=400px>
</p>

## Installation

Ibex can be installed through pip with
```bash
pip install prescient-ibex
```
Alternatively, you can use `uv` and create a new virtual environment
```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

## Usage

The simplest way to run inference is through the `ibex` command, e.g.

```bash
ibex --fv-heavy EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS --fv-light DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK --output prediction.pdb
```
You can provide a csv (with the `--csv` argument) or a parquet file (with the `--parquet` argument) and run a batched inference writing the output into a specified directory with
```bash
ibex --csv sequences.csv --output predictions
```
where `sequences.csv` should contain a `fv_heavy` and `fv_light` column with heavy and light chain sequences, and optionally an `id` column with a string that will be used as part of the output PDB filenames.

By default, structures are predicted in the holo conformation. To predict the apo state, use the `--apo` flag.

To run a refinement step on the predicted structures, use the `--refine` option. Additional checks to fix cis-isomers and D-stereoisomers during refinement can be activated with `--refine-checks`.
 
Instead of running Ibex, you can use `--abodybuilder3` to run inference with the [ABodyBuilder3](https://academic.oup.com/bioinformatics/article/40/10/btae576/7810444) model. 

To run Ibex programmatically, you can use
```python
from ibex import Ibex, checkpoint_path, inference
ckpt = checkpoint_path("ibex")
ibex_model = Ibex.load_from_ensemble_checkpoint(ckpt)
inference(ibex_model, fv_heavy, fv_light, "prediction.pdb")
```
to predict structures for multiple sequence pairs, `batch_inference` is recommended instead of `inference`.

## Predictions on nanobodies and TCRs

To predict nanobody structures, leave out the `fv_light` argument, or set it as `""` or `None` in the csv column. 

For inference on TCRs, you should provide the variable beta chain sequence as `fv_heavy` and the alpha chain as `fv_light`. Ibex has not been trained on gamma and delta chains.


## License
The codebase and the [ABodyBuilder3](https://academic.oup.com/bioinformatics/article/40/10/btae576/7810444) model weights are available under [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0). 

The [model weights](https://doi.org/10.5281/zenodo.15866555) for Ibex are available under [Genentech Apache 2.0 Non-Commercial license](https://github.com/prescient-design/ibex/blob/main/docs/Genentech_license_weights_ibex).

Ibex uses as input representation embeddings from ESMC 300M, which is licensed under the [EvolutionaryScale Cambrian Open License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement).

## Citation
If this code is useful to you please cite our paper using the following bibtex entry,

```bibtex
@article{ibex,
    author = {Dreyer et al.},
    title = "{Conformation-Aware Structure Prediction of Antigen-Recognizing Immune Proteins}",
    year = {2025},
}