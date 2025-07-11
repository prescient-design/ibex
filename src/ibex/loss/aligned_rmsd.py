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

import torch

CDR_RANGES_AHO = {
	"L1": (23,42),
	"L2": (56,72),
	"L3": (106,138),
	"H1": (23,42),
	"H2": (56,69),
	"H3": (106,138),
}

region_mapping = {
    "cdrh1": 0,
    "cdrh2": 1,
    "cdrh3": 2,
    "cdrl1": 3,
    "cdrl2": 4,
    "cdrl3": 5,
    "fwh1": 6,
    "fwh2": 7,
    "fwh3": 8,
    "fwh4": 9,
    "fwl1": 10,
    "fwl2": 11,
    "fwl3": 12,
    "fwl4": 13,
}

heavy_chain_regions = {0, 1, 2, 6, 7, 8, 9}
light_chain_regions = {3, 4, 5, 10, 11, 12, 13}

heavy_framework_regions = {6, 7, 8, 9}
light_framework_regions = {10, 11, 12, 13}

heavy_cdr_regions = {0, 1, 2}
light_cdr_regions = {3, 4, 5}


def apply_inverse_transformation(coords, R, t):
    # Apply inverse rotation and translation
    coords_transformed = torch.bmm(R.transpose(-1, -2), (coords - t.unsqueeze(1)).transpose(-1, -2)).transpose(-1, -2)
    return coords_transformed


def rmsd_summary_calculation(
    coords_truth: torch.Tensor,
    coords_prediction: torch.Tensor,
    sequence_mask: torch.Tensor,
    region_mask: torch.Tensor,
    chain_mask: torch.Tensor,
    batch_average: bool = True,
) -> dict[str, torch.Tensor]:
    """Computes RMSD summary for different regions and chains.

    Args:
        coords_truth (torch.Tensor): (B, n, 14/37, 3) ground truth coordinates
        coords_prediction (torch.Tensor): (B, n, 14/37, 3) predicted coordinates
        sequence_mask (torch.Tensor): (B, n) where [i, j] = 1 if a coordinate for sequence i at residue j exists.
        region_mask (torch.Tensor): (B, n) region mask indicating the region of each residue
        chain_mask (torch.Tensor): (B, n) chain mask indicating the chain of each residue (0 for light chain, 1 for heavy chain)
        batch_average (bool): if True, average along the batch dimensions

    Returns:
        dict[str, torch.Tensor]: RMSD values for each region and chain
    """
    results = {}

    def apply_transformation(coords, R, t):
        # Apply inverse rotation and translation
        coords_transformed = torch.bmm(R.transpose(-1, -2), (coords - t.unsqueeze(1)).transpose(-1, -2)).transpose(-1, -2)
        return coords_transformed

    # Align and compute RMSD for heavy chain regions
    heavy_chain_mask = chain_mask == 1

    heavy_chain_backbone_truth = extract_backbone_coordinates(
        coords_truth * heavy_chain_mask.unsqueeze(-1).unsqueeze(-1)
    )
    heavy_chain_sequence_mask = extract_backbone_mask(sequence_mask * heavy_chain_mask)

    # Mask for framework regions only
    heavy_framework_mask = (region_mask.unsqueeze(-1) == torch.tensor(list(heavy_framework_regions), device=region_mask.device)).any(-1) * heavy_chain_mask
    heavy_framework_backbone_truth = extract_backbone_coordinates(
        coords_truth * heavy_framework_mask.unsqueeze(-1).unsqueeze(-1)
    )
    heavy_framework_backbone_prediction = extract_backbone_coordinates(
        coords_prediction * heavy_framework_mask.unsqueeze(-1).unsqueeze(-1)
    )
    heavy_framework_sequence_mask = extract_backbone_mask(sequence_mask * heavy_framework_mask)

    # Align framework regions
    heavy_framework_backbone_truth, R, t = batch_align(
        heavy_framework_backbone_truth, heavy_framework_backbone_prediction, heavy_framework_sequence_mask, return_transform=True
    )

    # Compute RMSD for heavy chain framework as a whole
    square_distance = (
        torch.linalg.norm(
            heavy_framework_backbone_prediction - heavy_framework_backbone_truth, dim=-1
        )
        ** 2
    )
    square_distance = square_distance * heavy_framework_sequence_mask

    heavy_framework_msd = torch.sum(square_distance, dim=-1) / heavy_framework_sequence_mask.sum(dim=-1)
    heavy_framework_rmsd = torch.sqrt(heavy_framework_msd)

    if batch_average:
        heavy_framework_rmsd = heavy_framework_rmsd.mean()

    results["fwh_rmsd"] = heavy_framework_rmsd

    # Apply the same transformation to the CDR regions
    heavy_cdr_mask = (region_mask.unsqueeze(-1) == torch.tensor(list(heavy_cdr_regions), device=region_mask.device)).any(-1) * heavy_chain_mask
    heavy_cdr_backbone_prediction = extract_backbone_coordinates(
        coords_prediction * heavy_cdr_mask.unsqueeze(-1).unsqueeze(-1)
    )
    heavy_cdr_backbone_prediction_aligned = apply_transformation(heavy_cdr_backbone_prediction, R, t)

    for region_name, region_idx in region_mapping.items():
        if region_idx in heavy_cdr_regions:
            region_mask_region = region_mask == region_idx
            region_mask_backbone = extract_backbone_mask(region_mask_region)

            heavy_chain_region_mask = region_mask_backbone * heavy_chain_sequence_mask
            square_distance = (
                torch.linalg.norm(
                    heavy_cdr_backbone_prediction_aligned - heavy_chain_backbone_truth, dim=-1
                )
                ** 2
            )
            square_distance = square_distance * heavy_chain_region_mask

            region_msd = torch.sum(square_distance, dim=-1) / heavy_chain_region_mask.sum(dim=-1)
            region_rmsd = torch.sqrt(region_msd)

            if batch_average:
                region_rmsd = region_rmsd.mean()

            results[f"{region_name}_rmsd"] = region_rmsd

    # Align and compute RMSD for light chain regions
    light_chain_mask = chain_mask == 0

    light_chain_backbone_truth = extract_backbone_coordinates(
        coords_truth * light_chain_mask.unsqueeze(-1).unsqueeze(-1)
    )
    light_chain_sequence_mask = extract_backbone_mask(sequence_mask * light_chain_mask)

    # Mask for framework regions only
    light_framework_mask = (region_mask.unsqueeze(-1) == torch.tensor(list(light_framework_regions), device=region_mask.device)).any(-1) * light_chain_mask
    light_framework_backbone_truth = extract_backbone_coordinates(
        coords_truth * light_framework_mask.unsqueeze(-1).unsqueeze(-1)
    )
    light_framework_backbone_prediction = extract_backbone_coordinates(
        coords_prediction * light_framework_mask.unsqueeze(-1).unsqueeze(-1)
    )
    light_framework_sequence_mask = extract_backbone_mask(sequence_mask * light_framework_mask)

    # Align framework regions
    light_framework_backbone_truth, R, t = batch_align(
        light_framework_backbone_truth, light_framework_backbone_prediction, light_framework_sequence_mask, return_transform=True
    )

    # Compute RMSD for light chain framework as a whole
    square_distance = (
        torch.linalg.norm(
            light_framework_backbone_prediction - light_framework_backbone_truth, dim=-1
        )
        ** 2
    )
    square_distance = square_distance * light_framework_sequence_mask

    light_framework_msd = torch.sum(square_distance, dim=-1) / light_framework_sequence_mask.sum(dim=-1)
    light_framework_rmsd = torch.sqrt(light_framework_msd)

    if batch_average:
        light_framework_rmsd = light_framework_rmsd.mean()

    results["fwl_rmsd"] = light_framework_rmsd

    # Apply the same transformation to the CDR regions
    light_cdr_mask = (region_mask.unsqueeze(-1) == torch.tensor(list(light_cdr_regions), device=region_mask.device)).any(-1) * light_chain_mask
    light_cdr_backbone_prediction = extract_backbone_coordinates(
        coords_prediction * light_cdr_mask.unsqueeze(-1).unsqueeze(-1)
    )
    light_cdr_backbone_prediction_aligned = apply_transformation(light_cdr_backbone_prediction, R, t)

    for region_name, region_idx in region_mapping.items():
        if region_idx in light_cdr_regions:
            region_mask_region = region_mask == region_idx
            region_mask_backbone = extract_backbone_mask(region_mask_region)

            light_chain_region_mask = region_mask_backbone * light_chain_sequence_mask
            square_distance = (
                torch.linalg.norm(
                    light_cdr_backbone_prediction_aligned - light_chain_backbone_truth, dim=-1
                )
                ** 2
            )
            square_distance = square_distance * light_chain_region_mask

            region_msd = torch.sum(square_distance, dim=-1) / light_chain_region_mask.sum(dim=-1)
            region_rmsd = torch.sqrt(region_msd)

            if batch_average:
                region_rmsd = region_rmsd.mean()

            results[f"{region_name}_rmsd"] = region_rmsd

    return results


def aligned_fv_and_cdrh3_rmsd(
    coords_truth: torch.Tensor,
    coords_prediction: torch.Tensor,
    sequence_mask: torch.Tensor,
    cdrh3_mask: torch.Tensor,
    batch_average: bool = True,
) -> dict[str, torch.Tensor]:
    """Aligns positions_truth to positions_prediction in a batched way.

    Args:
        positions_truth (torch.Tensor): (B, n, 14/37, 3) ground truth coordinates
        positions_prediction (torch.Tensor): (B, n, 14/37, 3) predicted coordinates
        sequence_mask (torch.Tensor): (B, n) where [i, j] = 1 if a coordinate for sequence i at residue j exists.
        cdrh3_mask (torch.Tensor): (B, n) where [i, j] = 1 if a coordinate for sequence i at residue j is part of the cdrh3 loop.
        batch_average (bool): if True, average along the batch dimensions

    Returns:
        A dictionary[str, torch.Tensor] containing
            seq_rmsd: the RMSD of the backbone after backbone alignment
            cdrh3_rmsd: the RMSD of the CDRH3 backbone after backbone alignment
    """

    # extract backbones and mask and put in 3d point cloud shape
    backbone_truth = extract_backbone_coordinates(coords_truth)
    backbone_prediction = extract_backbone_coordinates(coords_prediction)
    backbone_sequence_mask = extract_backbone_mask(sequence_mask)

    # align backbones
    backbone_truth = batch_align(
        backbone_truth, backbone_prediction, backbone_sequence_mask
    )

    square_distance = (
        torch.linalg.norm(backbone_prediction - backbone_truth, dim=-1) ** 2
    )
    square_distance = square_distance * backbone_sequence_mask

    seq_msd = square_distance.sum(dim=-1) / backbone_sequence_mask.sum(dim=-1)
    seq_rmsd = torch.sqrt(seq_msd)

    backbone_cdrh3_mask = extract_backbone_mask(cdrh3_mask)
    square_distance = square_distance * (backbone_cdrh3_mask * backbone_sequence_mask)
    cdrh3_msd = torch.sum(square_distance, dim=-1) / backbone_cdrh3_mask.sum(dim=-1)
    cdrh3_rmsd = torch.sqrt(cdrh3_msd)

    if batch_average:
        seq_rmsd = seq_rmsd.mean()
        cdrh3_rmsd = cdrh3_rmsd.mean()

    return {"seq_rmsd": seq_rmsd, "cdrh3_rmsd": cdrh3_rmsd}


def extract_backbone_coordinates(positions: torch.Tensor) -> torch.Tensor:
    """(B, n, 14/37, 3) -> (B, n * 4, 3)"""
    batch_size = positions.size(0)
    backbone_positions = positions[:, :, :4, :]  # (B, n, 4, 3)
    backbone_positions_flat = backbone_positions.reshape(
        batch_size, -1, 3
    )  # (B, n * 4, 3)
    return backbone_positions_flat


def extract_backbone_mask(sequence_mask: torch.Tensor) -> torch.Tensor:
    """(B, n) -> (B, n * 4)"""
    batch_size = sequence_mask.size(0)
    return sequence_mask.unsqueeze(-1).repeat(1, 1, 4).view(batch_size, -1)


def batch_align(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, return_transform=False):
    """Aligns 3-dimensional point clouds. Based on section 4 of https://igl.ethz.ch/projects/ARAP/svd_rot.pdf.

    Args:
        x (torch.Tensor): A tensor shape (B, n, 3)
        y (torch.Tensor): A tensor shape (B, n, 3)
        mask (torch.Tensor): A mask of shape (B, n) were mask[i, j]=1 indicates the presence of a point in sample i at location j of both sequences.
        return_transform (bool): If True, return rotation and translation matrices.

    Returns:
        torch.Tensor: a rototranslated x aligned to y.
        torch.Tensor: rotation matrix used for alignment (if return_transform is True).
        torch.Tensor: translation matrix used for alignment (if return_transform is True).
    """

    # check inputs
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim=3. Instead got {x.ndim=}")
    if y.ndim != 3:
        raise ValueError(f"Expected y.ndim=3. Instead got {x.ndim=}")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask.ndim=2. Instead got {mask.ndim=}")
    if x.size(-1) != 3:
        raise ValueError(f"Expected last dim of x to be 3. Instead got {x.size(-1)=}")
    if y.size(-1) != 3:
        raise ValueError(f"Expected last dim of y to be 3. Instead got {y.size(-1)=}")

    # (B, n) -> (B, n, 1)
    mask = mask.unsqueeze(-1)

    # zero masked coordinates (the below centroids computation relies on it).
    x = x * mask
    y = y * mask

    # centroids (B, 3)
    p_bar = x.sum(dim=1) / mask.sum(dim=1)
    q_bar = y.sum(dim=1) / mask.sum(dim=1)

    # centered points (B, n, 3)
    x_centered = x - p_bar.unsqueeze(1)
    y_centered = y - q_bar.unsqueeze(1)

    # compute covariance matrices (B, 3, 3)
    num_valid_points = mask.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
    S = torch.bmm(x_centered.transpose(-1, -2), y_centered * mask) / num_valid_points
    S = S + 10e-6 * torch.eye(S.size(-1)).unsqueeze(0).to(S.device)

    # Compute U, V from SVD (B, 3, 3)
    U, _, Vh = torch.linalg.svd(S)
    V = Vh.transpose(-1, -2)
    Uh = U.transpose(-1, -2)

    # correction that accounts for reflection (B, 3, 3)
    correction = torch.eye(x.size(-1)).unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)
    correction[:, -1, -1] = torch.det(torch.bmm(V, Uh).float())

    # rotation (B, 3, 3)
    R = V.bmm(correction).bmm(Uh)

    # translation (B, 3)
    t = q_bar - R.bmm(p_bar.unsqueeze(-1)).squeeze()

    # translate x to align with y
    x_rotated = torch.bmm(R, x.transpose(-1, -2)).transpose(-1, -2)
    x_aligned = x_rotated + t.unsqueeze(1)

    if return_transform:
        return x_aligned, R, t
    else:
        return x_aligned

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from ibex.dataloader import ABDataset, collate_fn
    dataset = ABDataset("test", split_file="/data/dreyerf1/ibex/split.csv", data_dir="/data/dreyerf1/ibex/structures", edge_chain_feature=True)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)
    for batch in dataloader:
        coords = batch["atom14_gt_positions"]
        preds = coords + 10
        mask = batch["seq_mask"]
        cdrh3_mask = batch["region_numeric"] == 2
        print(aligned_fv_and_cdrh3_rmsd(coords, preds, mask, cdrh3_mask))
        break

