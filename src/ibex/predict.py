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


from pathlib import Path
from typing import Optional
import torch
import tempfile
from tqdm import tqdm
from loguru import logger

from ibex.model import Ibex, EnsembleStructureModule
from ibex.refine import refine_file


def process_file(pdb_string, output_file, refine, refine_checks=False):
    if not refine:
        with open(output_file, "w") as f:
            f.write(pdb_string)
    else:
        try:
            with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
                tmpfile.write(pdb_string.encode('utf-8'))
                tmpfile.flush()
                refine_file(tmpfile.name, output_file, checks=refine_checks)
        except Exception as e:
            logger.warning(f"Refinement failed with error: {e}")
            with open(output_file, "w") as f:
                f.write(pdb_string)


def inference(
    model: Ibex,
    fv_heavy: str,
    fv_light: str,
    output_file: Path,
    logging: bool = True,
    save_all = False,
    refine: bool = False,
    refine_checks: bool = False,
    apo: bool = False,
    return_pdb: bool = True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and logging:
        logger.warning("Inference is being done on CPU as GPU not found.")
    if save_all==True and not isinstance(model.model, EnsembleStructureModule):
        raise ValueError("save_all is set to True but model is not an ensemble model.")
    if return_pdb==False and refine:
        raise ValueError("Cannot return a protein object and refine at the same time. To run refinement, output format must be a PDB file (return_pdb==True).")
    if return_pdb==False and save_all:
        raise ValueError("Cannot return a protein object and save all outputs at the same time. To save all, output format must be a PDB file (return_pdb==True).")
    with torch.no_grad():
        pdb_string_or_protein = model.predict(fv_heavy, fv_light, device=device, ensemble=save_all, pdb_string=return_pdb, apo=apo)
        if not return_pdb:
            if logging:
                logger.info("Inference complete. Returning a protein object.")
            return pdb_string_or_protein
        if save_all:
            ensemble_files = []
            for i, pdb_string_current in enumerate(pdb_string_or_protein):
                output_file_current = output_file.parent / f"{output_file.stem}_{i+1}{output_file.suffix}"
                process_file(pdb_string_current, output_file_current, refine, refine_checks)
                ensemble_files.append(str(output_file_current))
            output_file = ensemble_files
        else:
            process_file(pdb_string_or_protein, output_file, refine, refine_checks)
    if logging:
        logger.info(f"Inference complete. Wrote PDB file to {output_file=}")
    return output_file


def batch_inference(
    model: Ibex,
    fv_heavy_list: list[str],
    fv_light_list: list[str],
    output_dir: Path,
    batch_size: int,
    output_names: Optional[list[str]] = None,
    logging: bool = True,
    refine: bool = False,
    refine_checks: bool = False,
    apo_list: bool = None,
    return_pdb: bool = True
):
    if output_names is None:
        output_names = [f"output_{i}" for i in range(len(fv_heavy_list))]

    if len(fv_heavy_list) != len(fv_light_list) or len(fv_heavy_list) != len(output_names):
        raise ValueError("Input lists must have the same length.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and logging:
        logger.warning("Inference is being done on CPU as GPU not found.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if apo_list is not None and not model.conformation_aware:
        raise ValueError("Model is not conformation-aware, but apo_list was provided.")
    if return_pdb==False and refine:
        raise ValueError("Cannot return a protein object and refine at the same time. To run refinement, output format must be a PDB file (return_pdb==True).")

    if model.plm_model is not None:
        model.plm_model = model.plm_model.to(device)

    name_idx = 0  # Index for tracking the position in output_names
    result_files = []
    for i in tqdm(range(0, len(fv_heavy_list), batch_size), desc="Processing batches"):
        fv_heavy_batch = fv_heavy_list[i:i+batch_size]
        fv_light_batch = fv_light_list[i:i+batch_size] if fv_light_list else None
        with torch.no_grad():
            pdb_strings_or_proteins = model.predict_batch(
                fv_heavy_batch, fv_light_batch, device=device, pdb_string=return_pdb, apo_list=apo_list
            )
            if not return_pdb:
                if logging:
                    logger.warning("Inference complete. Returning a protein object.")
                return pdb_strings_or_proteins
            for pdb_string in pdb_strings_or_proteins:
                output_file = output_dir / f"{output_names[name_idx]}.pdb"
                process_file(pdb_string, output_file, refine, refine_checks)
                result_files.append(output_file)
                name_idx += 1
    if logging:
        logger.info(f"Inference complete. Wrote {name_idx} PDB files to {output_dir=}")

    return result_files
