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
import io

import numpy as np
import torch
from typing import MutableMapping
from omegaconf import DictConfig

from ibex.openfold.utils.data_transforms import make_atom14_masks
from ibex.openfold.utils.protein import Protein, to_pdb
from ibex.openfold.utils.feats import atom14_to_atom37


def compute_plddt(plddt: torch.Tensor) -> torch.Tensor:
    """Computes plddt from the model output. The output is a histogram of unnormalised
    plddt.

    Args:
        plddt (torch.Tensor): (B, n, 50) output from the model

    Returns:
        torch.Tensor: (B, n) plddt scores
    """
    pdf = torch.nn.functional.softmax(plddt, dim=-1)
    vbins = torch.arange(1, 101, 2).to(plddt.device).float()
    output = pdf @ vbins  # (B, n)
    return output


def add_atom37_to_output(output: dict, aatype: torch.Tensor):
    """Adds atom37 coordinates to an output dictionary containing atom14 coordinates."""
    atom14 = output["positions"][-1, 0]
    batch = make_atom14_masks({"aatype": aatype.squeeze()})
    atom37 = atom14_to_atom37(atom14, batch)
    output["atom37"] = atom37
    output["atom37_atom_exists"] = batch["atom37_atom_exists"]
    return output


def output_to_protein(output: dict, model_input: dict) -> Protein:
    """Generates a Protein object from Ibex predictions.

    Args:
        output (dict): Ibex output dictionary
        model_input (dict): Ibex input dictionary

    Returns:
        str: the contents of a pdb file in string format.
    """
    aatype = model_input["aatype"].squeeze().cpu().numpy().astype(int)
    atom37 = output["atom37"]
    chain_index = 1 - model_input["is_heavy"].cpu().numpy().astype(int)
    atom_mask = output["atom37_atom_exists"].cpu().numpy().astype(int)
    residue_index = np.arange(len(atom37))
    if "plddt" in output:
        plddt = compute_plddt(output["plddt"]).squeeze().detach().cpu().numpy()
        b_factors = np.expand_dims(plddt, 1).repeat(37, 1)
    else:
        b_factors = np.zeros_like(atom_mask)
    protein = Protein(
        aatype=aatype,
        atom_positions=atom37,
        atom_mask=atom_mask,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index,
    )

    return protein

def output_to_pdb(output: dict, model_input: dict) -> str:
    """Generates a pdb file from Ibex predictions.

    Args:
        output (dict): Ibex output dictionary
        model_input (dict): Ibex input dictionary

    Returns:
        str: the contents of a pdb file in string format.
    """
    return to_pdb(output_to_protein(output, model_input))
