# Copyright 2025 Genentech
# Copyright 2024 Exscientia
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

# Based on AlphaFoldLoss class from https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py#L1685

import loguru
import ml_collections
import torch
import torch.nn.functional as F
from loguru import logger

from ibex.loss.aligned_rmsd import aligned_fv_and_cdrh3_rmsd
from ibex.openfold.utils.loss import (
    compute_renamed_ground_truth,
    fape_loss,
    final_output_backbone_loss,
    find_structural_violations,
    lddt_loss,
    supervised_chi_loss,
    violation_loss_bondangle,
    violation_loss_bondlength,
    violation_loss_clash,
)


def triplet_loss(positive, negative, anchor, margin):
        # Compute cosine similarity
        sim_pos = F.cosine_similarity(anchor, positive, dim=-1)
        sim_neg = F.cosine_similarity(anchor, negative, dim=-1)
        # Triplet loss
        loss_triplet = torch.mean(torch.clamp(margin - sim_pos + sim_neg, min=0.0))
        return loss_triplet


def triplet_loss_batch(single: torch.Tensor, cdr_mask: torch.Tensor, margin: float=1.0):
    cdr_mask_expanded = cdr_mask.unsqueeze(-1).expand_as(single)
    masked_embeddings = single * cdr_mask_expanded
    sum_embeddings = masked_embeddings.sum(dim=1)
    count_non_zero = cdr_mask.sum(dim=1).unsqueeze(-1)
    # Avoid division by zero
    count_non_zero = torch.where(count_non_zero == 0, torch.ones_like(count_non_zero), count_non_zero)
    # Compute the average
    average_embeddings = sum_embeddings / count_non_zero
    loss = triplet_loss(average_embeddings[-3], average_embeddings[-2], average_embeddings[-1], margin)
    loss += triplet_loss(average_embeddings[-6], average_embeddings[-5], average_embeddings[-4], margin)
    return loss


def negative_pairs_loss(positive: torch.Tensor, negative: torch.Tensor, threshold: float):
        cosine_sim = F.cosine_similarity(positive, negative, dim=-1)
        # Compute loss
        loss = torch.mean(torch.clamp(threshold - cosine_sim, min=0.0))
        return loss

def negative_pairs_loss_batch(single: torch.Tensor, cdr_mask: torch.Tensor, threshold: float, pair_mask: torch.Tensor):
    if torch.tensor(pair_mask).sum() == 0:
        return torch.tensor(0.0, device=single.device)
    cdr_mask_expanded = cdr_mask.unsqueeze(-1).expand_as(single)
    masked_embeddings = single * cdr_mask_expanded
    sum_embeddings = masked_embeddings.sum(dim=1)
    count_non_zero = cdr_mask.sum(dim=1).unsqueeze(-1)
    # Avoid division by zero
    count_non_zero = torch.where(count_non_zero == 0, torch.ones_like(count_non_zero), count_non_zero)
    # Compute the average
    average_embeddings = sum_embeddings / count_non_zero
    paired_samples = average_embeddings[pair_mask]
    pos_samples = paired_samples[0::2]
    neg_samples = paired_samples[1::2]
    loss = negative_pairs_loss(pos_samples, neg_samples, threshold)
    return loss

class IbexLoss(torch.nn.Module):
    def __init__(self, config: ml_collections.config_dict.ConfigDict):
        super().__init__()
        self.config = config
        self.dist_and_angle_annealing = 0.0

    def forward(self, output: dict, batch: dict, finetune: bool = False, contrastive: bool = False):
        if finetune:
            output["violation"] = find_structural_violations(
                batch,
                output["positions"][-1],
                **self.config.violation,
            )

        if "renamed_atom14_gt_positions" not in output.keys():
            batch.update(
                compute_renamed_ground_truth(
                    batch,
                    output["positions"][-1],
                )
            )

        loss_fns = {
            "fape": lambda: fape_loss(
                {"sm": output},
                batch,
                self.config.fape,
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                output["angles"],
                output["unnormalized_angles"],
                **{**batch, **self.config.supervised_chi},
            ),
            "final_output_backbone_loss": lambda: final_output_backbone_loss(
                output, batch
            ),
        }
        if "plddt" in output:
            loss_fns.update(
                {
                    "plddt": lambda: lddt_loss(
                        output["plddt"],
                        output["positions"][-1],
                        batch["atom14_gt_positions"],
                        batch["atom14_atom_exists"],
                        batch["resolution"],
                    ),
                }
            )

        if finetune:
            loss_fns.update(
                {
                    "violation_loss_bondlength": lambda: violation_loss_bondlength(
                        output["violation"]
                    ),
                    "violation_loss_bondangle": lambda: violation_loss_bondangle(
                        output["violation"]
                    ),
                    "violation_loss_clash": lambda: violation_loss_clash(
                        output["violation"], **batch
                    ),
                }
            )
        if contrastive:
            loss_fns.update(
                {
                    "contrastive": lambda: negative_pairs_loss_batch(
                        output["single"],
                        batch["cdr_mask"],
                        self.config.contrastive.margin,
                        batch["is_matched"],
                    ),
                    # "contrastive": lambda: triplet_loss_batch(
                    #     output["single"],
                    #     batch["cdr_mask"],
                    #     self.config.contrastive.margin,
                    # ),
                }
            )
        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0.0, requires_grad=True)
            if loss_name in ["violation_loss_bondlength", "violation_loss_bondangle"]:
                weight *= min(self.dist_and_angle_annealing / 50, 1)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
            losses[f"{loss_name}_weighted"] = weight * losses[loss_name]
            losses[f"{loss_name}_weight"] = weight

        # aligned_rmsd (not added to cum_loss)
        with torch.no_grad():
            losses.update(
                aligned_fv_and_cdrh3_rmsd(
                    coords_truth=batch["atom14_gt_positions"],
                    coords_prediction=output["positions"][-1],
                    sequence_mask=batch["seq_mask"],
                    cdrh3_mask=batch["region_numeric"] == 2,
                )
            )

        losses["loss"] = cum_loss.detach().clone()

        return cum_loss, losses
