# Copyright 2025 Genentech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from ibex.model import Ibex
from ibex.predict import MODEL_CHECKPOINTS, ENSEMBLE_MODELS, checkpoint_path, inference, batch_inference

warnings.filterwarnings("ignore")


def main(
    abodybuilder3: bool = typer.Option(False, help="Use the AbodyBuilder3 model instead of Ibex for inference."),
    ckpt: str = typer.Option("", help="Path to model checkpoint. This is only needed to load a user specified checkpoint."),
    fv_heavy: str = typer.Option("", help="Sequence of the heavy chain."),
    fv_light: str = typer.Option("", help="Sequence of the light chain."),
    csv: str = typer.Option("", help="CSV file containing sequences of heavy and light chains. Columns should be named 'fv_heavy' and 'fv_light'. Output file names can be provided in a 'id' column."),
    parquet: str = typer.Option("", help="Parquet file containing sequences of heavy and light chains. Columns should be named 'fv_heavy' and 'fv_light'. Output file names can be provided in a 'id' column."),
    output: Path = typer.Option("prediction.pdb", help="Output file for the PDB structure, or path to the output folder when a parquet or csv file is provided."),
    batch_size: int = typer.Option(32, help="Batch size for inference if a parquet or csv file is provided."),
    ensemble: bool = typer.Option(False, help="Specify if checkpoint provided is an ensemble (only needed if using the explicit --ckpt flag)."),
    save_all: bool = typer.Option(False, help="Save all structures of the ensemble as output files."),
    refine: bool = typer.Option(False, help="Refine the output structures with openMM."),
    refine_checks: bool = typer.Option(False, help="Additional checks to fix cis-isomers and D-stereoisomers during refinement."),
    apo: bool = typer.Option(False, help="Predict structures in the apo conformation."),
):
    """
    Model can be specified by name or by providing a checkpoint path. If a CSV or parquet file is provided, the model will perform inference on all sequences in the file, otherwise it will perform inference on a single heavy and light sequence pair.
    """
    model = "ibex"
    if abodybuilder3:
        model = "abodybuilder3"
    if ckpt=="" and model not in MODEL_CHECKPOINTS:
        typer.echo(f"Invalid model name: {model}. Valid options are: {', '.join(MODEL_CHECKPOINTS.keys())} or provide a checkpoint path with the --ckpt option.")
        raise typer.Exit(code=1)
    if ckpt == "":
        ckpt = checkpoint_path(model)
    ckpt=Path(ckpt)

    if ensemble or model in ENSEMBLE_MODELS:
        logger.info(f"Loading ensemble model from {ckpt=}")
        ibex_model = Ibex.load_from_ensemble_checkpoint(ckpt)
    else:
        logger.info(f"Loading single model from {ckpt=}")
        ibex_model = Ibex.load_from_checkpoint(ckpt)
        if ibex_model.plm_model is not None:
            ibex_model.set_plm()
    if fv_heavy:
        logger.info("Performing inference on a single heavy and light sequence pair.")
        inference(ibex_model, fv_heavy, fv_light, output, save_all=save_all, refine=refine, refine_checks=refine_checks, apo=apo)
    elif csv or parquet:
        if save_all:
            logger.warning("save_all was set to True, but ensemble output is not implemented for batched inference. Setting save_all to False.")
        if output==Path("prediction.pdb"):
            # overwrite default for batch inference
            output = Path("predictions")
        if csv:
            csv = Path(csv)
            logger.info(f"Performing inference on sequences from {csv=}")
            df = pd.read_csv(csv)
        else:
            parquet = Path(parquet)
            logger.info(f"Performing inference on sequences from {parquet=}")
            df = pd.read_parquet(parquet)
        fv_heavy_list = df["fv_heavy"].tolist()
        fv_light_list = df["fv_light"].tolist()
        apo_list = None
        if ibex_model.conformation_aware:
            apo_list = [apo]*len(fv_heavy_list)
        if "id" in df.columns:
            names = df["id"].tolist()
            batch_inference(
                ibex_model,
                fv_heavy_list,
                fv_light_list,
                output,
                batch_size,
                names,
                refine=refine,
                refine_checks=refine_checks,
                apo_list=apo_list
            )
        else:
            batch_inference(
                ibex_model,
                fv_heavy_list,
                fv_light_list,
                output,
                batch_size,
                refine=refine,
                refine_checks=refine_checks,
                apo_list=apo_list
            )
    else:
        typer.echo("Please provide sequences of heavy and light chains or a csv/parquet file containing sequences.")
        raise typer.Exit(code=1)


def app():
    typer.run(main)
