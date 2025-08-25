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
from ibex.loss.aligned_rmsd import positions_to_backbone_dihedrals, region_mapping, CDR_RANGES_AHO


def dihedral_distance_per_loop(
    positions_predicted: torch.Tensor,
    positions_reference: torch.Tensor, 
    region_mask: torch.Tensor, 
    mask: torch.Tensor | None = None,
    residue_index: torch.Tensor | None = None, 
    chain_index: torch.Tensor | None = None
) -> dict[str, float]:
    """Computes the dihedral distance between predicted and reference backbone coordinates for each loop region.
    
    Args:
        positions_predicted (torch.Tensor): Predicted atomic coordinates of the protein backbone.
        positions_reference (torch.Tensor): Reference (ground truth) atomic coordinates of the protein backbone.
        region_mask (torch.Tensor): Boolean mask specifying residues belonging to each loop region (e.g., CDRs).
        mask (torch.Tensor | None, optional): Boolean mask indicating valid residues in the sequence.
        residue_index (torch.Tensor | None, optional): Indices of residues in the sequence. Defaults to None.
        chain_index (torch.Tensor | None, optional): Indices of chains in the structure. Defaults to None.
    Returns:
        dict[str, float]: Dictionary mapping each loop region to its computed dihedral distance between predicted and reference structures.
    
    """
    if mask is None:
        mask = torch.ones(positions_reference.shape[:-1]).to(device=positions_reference.device)
    dihedral_angles_predicted, dihedral_mask = positions_to_backbone_dihedrals(positions_predicted, mask, residue_index, chain_index)
    dihedral_angles_reference, _ = positions_to_backbone_dihedrals(positions_reference, mask, residue_index, chain_index)
    results = {}
    dihedral_differences = 2 * (1 - torch.cos(dihedral_angles_predicted - dihedral_angles_reference))
    for region_name, region_idx in region_mapping.items():
        if region_name.startswith("cdr") and (region_mask == region_idx).any():
            results[region_name] = torch.mean(dihedral_differences[region_mask == region_idx]).item()
    return results

def region_mask_from_aho(fv_heavy_aho: str, fv_light_aho: str = "") -> torch.Tensor:
    """Return a tensor with CDR and framework identifiers for a given aho string of heavy and light chains.
    Args:
        fv_heavy_aho (str): Heavy chain aho string
        fv_light_aho (str): Light chain aho string
    Returns:
        torch.Tensor: Tensor of shape [N]
    """
    region_len = {}
    region_len["fwh1"]  = len(fv_heavy_aho[:CDR_RANGES_AHO["H1"][0]].replace('-', ''))
    region_len["cdrh1"] = len(fv_heavy_aho[CDR_RANGES_AHO["H1"][0]:CDR_RANGES_AHO["H1"][1]].replace('-', ''))
    region_len["fwh2"]  = len(fv_heavy_aho[CDR_RANGES_AHO["H1"][1]:CDR_RANGES_AHO["H2"][0]].replace('-', ''))
    region_len["cdrh2"] = len(fv_heavy_aho[CDR_RANGES_AHO["H2"][0]:CDR_RANGES_AHO["H2"][1]].replace('-', ''))
    region_len["fwh3"]  = len(fv_heavy_aho[CDR_RANGES_AHO["H2"][1]:CDR_RANGES_AHO["H3"][0]].replace('-', ''))
    region_len["cdrh3"] = len(fv_heavy_aho[CDR_RANGES_AHO["H3"][0]:CDR_RANGES_AHO["H3"][1]].replace('-', ''))
    region_len["fwh4"]  = len(fv_heavy_aho[CDR_RANGES_AHO["H3"][1]:].replace('-', ''))

    if fv_light_aho:
        region_len["fwl1"]  = len(fv_light_aho[:CDR_RANGES_AHO["L1"][0]].replace('-', ''))
        region_len["cdrl1"] = len(fv_light_aho[CDR_RANGES_AHO["L1"][0]:CDR_RANGES_AHO["L1"][1]].replace('-', ''))
        region_len["fwl2"]  = len(fv_light_aho[CDR_RANGES_AHO["L1"][1]:CDR_RANGES_AHO["L2"][0]].replace('-', ''))
        region_len["cdrl2"] = len(fv_light_aho[CDR_RANGES_AHO["L2"][0]:CDR_RANGES_AHO["L2"][1]].replace('-', ''))
        region_len["fwl3"]  = len(fv_light_aho[CDR_RANGES_AHO["L2"][1]:CDR_RANGES_AHO["L3"][0]].replace('-', ''))
        region_len["cdrl3"] = len(fv_light_aho[CDR_RANGES_AHO["L3"][0]:CDR_RANGES_AHO["L3"][1]].replace('-', ''))
        region_len["fwl4"]  = len(fv_light_aho[CDR_RANGES_AHO["L3"][1]:].replace('-', ''))
        

    res = []
    for region in region_len:
        res.append(torch.ones(region_len[region], dtype=torch.int) * region_mapping[region])
    return torch.cat(res)

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
