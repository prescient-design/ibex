[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15866555.svg)](https://doi.org/10.5281/zenodo.15866555)

# Ibex 🐐

[Ibex](https://arxiv.org/abs/2507.09054) is a lightweight antibody and TCR structure prediction model.

<p align="center">
<img src="https://raw.githubusercontent.com/prescient-design/ibex/refs/heads/main/docs/assets/ibex.png" width=400px>
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
Below is a summary of all available options:
```
--abodybuilder3    Use the AbodyBuilder3 model instead of Ibex for inference. [default: no-abodybuilder3]                                           
--fv-heavy         Sequence of the heavy chain.                                                                                                     
--fv-light         Sequence of the light chain.                                                                                                     
--csv              CSV file containing sequences of heavy and light chains. Columns should be named 'fv_heavy' and 'fv_light'. Output file names can
                   be provided in a 'id' column.                                                                                                    
--parquet          Parquet file containing sequences of heavy and light chains. Columns should be named 'fv_heavy' and 'fv_light'. Output file names
                   can be provided in a 'id' column.                                                                                                
--output           Output file for the PDB structure, or path to the output folder when a parquet or csv file is provided. [default: prediction.pdb]
--batch-size       Batch size for inference if a parquet or csv file is provided. [default: 32]                                                     
--save-all         Save all structures of the ensemble as output files. [default: no-save-all]                                                      
--refine           Refine the output structures with openMM. [default: no-refine]                                                                   
--refine-checks    Additional checks to fix cis-isomers and D-stereoisomers during refinement. [default: no-refine-checks]                          
--apo              Predict structures in the apo conformation. [default: no-apo]                                                                    
```

To run Ibex programmatically, you can use
```python
from ibex import Ibex, inference
ibex_model = Ibex.from_pretrained("ibex") # or "abodybuilder3"
inference(ibex_model, fv_heavy, fv_light, "prediction.pdb")
```
to predict structures for multiple sequence pairs, `batch_inference` is recommended instead of `inference`.

## Predictions on nanobodies and TCRs

To predict nanobody structures, leave out the `fv_light` argument, or set it as `""` or `None` in the csv column. 

For inference on TCRs, you should provide the variable beta chain sequence as `fv_heavy` and the alpha chain as `fv_light`. Ibex has not been trained on gamma and delta chains.


## License
The Ibex codebase is available under an [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0), and the [ABodyBuilder3](https://doi.org/10.5281/zenodo.11354576) model weights under a [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/legalcode), both of which allow for commercial use.

The [Ibex model weights](https://doi.org/10.5281/zenodo.15866555) are available under a [Genentech Apache 2.0 Non-Commercial license](https://raw.githubusercontent.com/prescient-design/ibex/refs/heads/main/docs/Genentech_license_weights_ibex), which allows its use for non-commercial academic research purposes.

Ibex uses as input representation embeddings from ESMC 300M, which is licensed under the [EvolutionaryScale Cambrian Open License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement).

## Citation
When using Ibex in your work, please cite the following paper

```bibtex
@misc{ibex,
      title={Conformation-Aware Structure Prediction of Antigen-Recognizing Immune Proteins},
      author={Frédéric A. Dreyer and Jan Ludwiczak and Karolis Martinkus and Brennan Abanades and Robert G. Alberstein and Pan Kessel and Pranav Rao and Jae Hyeon Lee and Richard Bonneau and Andrew M. Watkins and Franziska Seeger},
      year={2025},
      eprint={2507.09054},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2507.09054},
}
```

If you use the ABodyBuilder3 model weights, you should also cite
```bibtex
@article{abodybuilder3,
    author = {Kenlay, Henry and Dreyer, Frédéric A and Cutting, Daniel and Nissley, Daniel and Deane, Charlotte M},
    title = "{ABodyBuilder3: improved and scalable antibody structure predictions}",
    journal = {Bioinformatics},
    volume = {40},
    number = {10},
    pages = {btae576},
    year = {2024},
    month = {10},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btae576}
}
```