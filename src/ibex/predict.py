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

import os
import requests
import hashlib
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
from loguru import logger
import tempfile
from tqdm import tqdm

from ibex.model import Ibex, EnsembleStructureModule
from ibex.refine import refine_file

MODEL_CHECKPOINTS = {
    "ibex": "https://zenodo.org/records/15866556/files/ibex_v1.ckpt",
    "abodybuilder3": "https://zenodo.org/records/15866556/files/abb3.ckpt",
}
ENSEMBLE_MODELS = ["ibex"]

def download_from_url(url, local_path):
    """Downloads a file from a URL with a progress bar."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(local_path)}") as progress_bar:
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(block_size):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
            
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                raise IOError("ERROR: Something went wrong during download")
                
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        # Clean up partially downloaded file
        if os.path.exists(local_path):
            os.remove(local_path)
        raise

def get_checkpoint_path(checkpoint_url, cache_dir=None):
    """Generates a unique local path for a given URL."""
    cache_dir = cache_dir or os.path.join(Path.home(), ".cache/ibex")
    os.makedirs(cache_dir, exist_ok=True)
    url_filename = os.path.basename(checkpoint_url)
    url_hash = hashlib.md5(checkpoint_url.encode()).hexdigest()[:8]
    filename = f"{url_hash}_{url_filename}"
    return os.path.join(cache_dir, filename)


def checkpoint_path(model_name, cache_dir=None):
    """
    Ensures the model checkpoint exists locally, downloading it if necessary.
    """
    if model_name not in MODEL_CHECKPOINTS:
        raise ValueError(f"Invalid model name: {model_name}")

    checkpoint_url = MODEL_CHECKPOINTS[model_name]
    local_path = get_checkpoint_path(checkpoint_url, cache_dir)

    if not os.path.exists(local_path):
        logger.info(f"Downloading checkpoint from {checkpoint_url} to {local_path}")
        # Call the new download function
        download_from_url(checkpoint_url, local_path)
        
    return local_path


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
    apo: bool = False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and logging:
        logger.warning("Inference is being done on CPU as GPU not found.")
    if save_all==True and not isinstance(model.model, EnsembleStructureModule):
        logger.warning("save_all is set to True but model is not an ensemble model.")
    with torch.no_grad():
        pdb_string = model.predict(fv_heavy, fv_light, device=device, ensemble=save_all, apo=apo)
        if save_all:
            ensemble_files = []
            for i, pdb_string_current in enumerate(pdb_string):
                output_file_current = output_file.parent / f"{output_file.stem}_{i+1}{output_file.suffix}"
                process_file(pdb_string_current, output_file_current, refine, refine_checks)
                ensemble_files.append(str(output_file_current))
            output_file = ensemble_files
        else:
            process_file(pdb_string, output_file, refine, refine_checks)
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
    apo_list: bool = None
):
    if output_names is None:
        output_names = [f"output_{i}" for i in range(len(fv_heavy_list))]

    if len(fv_heavy_list) != len(fv_light_list) or len(fv_heavy_list) != len(output_names):
        raise ValueError("Input lists must have the same length.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and logging:
        logger.warning("Inference is being done on CPU as GPU not found.")

    output_dir.mkdir(parents=True, exist_ok=True)

    if apo_list is not None and not model.conformation_aware:
        raise ValueError("Model is not conformation-aware, but apo_list was provided.")
    elif apo_list is None and model.conformation_aware:
        raise ValueError("Model is conformation-aware, but apo_list was not provided.")

    if model.plm_model is not None:
        model.plm_model = model.plm_model.to(device)

    name_idx = 0  # Index for tracking the position in output_names
    result_files = []
    for i in tqdm(range(0, len(fv_heavy_list), batch_size), desc="Processing batches"):
        fv_heavy_batch = fv_heavy_list[i:i+batch_size]
        fv_light_batch = fv_light_list[i:i+batch_size] if fv_light_list else None
        with torch.no_grad():
            pdb_strings = model.predict_batch(
                fv_heavy_batch, fv_light_batch, device=device, apo_list=apo_list
            )

            for pdb_string in pdb_strings:
                output_file = output_dir / f"{output_names[name_idx]}.pdb"
                process_file(pdb_string, output_file, refine, refine_checks)
                result_files.append(output_file)
                name_idx += 1
    if logging:
        logger.info(f"Inference complete. Wrote {name_idx} PDB files to {output_dir=}")

    return result_files
