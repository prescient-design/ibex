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


def apply_transformation(coords, R, t):
    # Apply inverse rotation and translation
    coords_transformed = torch.bmm(R.transpose(-1, -2), (coords - t.unsqueeze(1)).transpose(-1, -2)).transpose(-1, -2)
    return coords_transformed


def coordinates_to_dihedral(input: torch.Tensor) -> torch.Tensor:
    """Compute dihedral angle from a set of four points.

    Given an input tensor with shape (*, 4, 3) representing points (p1, p2, p3, p4)
    compute the dihedral angle between the plane defined by (p1, p2, p3) and (p2, p3, p4).

    Parameters
    ----------
    input: torch.Tensor
        Shape (*, 4, 3)

    Returns
    -------
    torsion: torch.Tensor
        Shape (*,)
    """
    assert input.ndim >= 3
    assert input.shape[-2] == 4
    assert input.shape[-1] == 3

    # difference vectors: [a = p2 - p1, b = p3 - p2, c = p4 - p3]
    delta = input[..., 1:, :] - input[..., :-1, :]
    a, b, c = torch.unbind(delta, dim=-2)

    # torsion angle is angle from axb to bxc counterclockwise around b
    # see https://www.math.fsu.edu/~quine/MB_10/6_torsion.pdf

    axb = torch.cross(a, b, dim=-1)
    bxc = torch.cross(b, c, dim=-1)

    # orthogonal basis in plane perpendicular to b
    # NOTE v1 and v2 are not unit but have the same magnitude
    v1 = axb
    v2 = torch.cross(torch.nn.functional.normalize(b, dim=-1), axb, dim=-1)

    x = torch.sum(bxc * v1, dim=-1)
    y = torch.sum(bxc * v2, dim=-1)
    phi = torch.atan2(y, x)

    return phi


def _positions_to_phi(
    positions: torch.Tensor,
    mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
):
    chain_boundary_mask = torch.cat(
        [torch.zeros_like(chain_index[..., :1], dtype=torch.bool), torch.diff(chain_index, n=1, dim=-1) == 0], dim=-1
    )

    chain_break_mask = torch.cat(
        [torch.zeros_like(residue_index[..., :1], dtype=torch.bool), torch.diff(residue_index, n=1, dim=-1) == 1],
        dim=-1,
    )

    (input, mask) = (
        torch.stack(
            [
                x[..., :-1, 2, :],  # C(i-1)
                x[..., 1:, 0, :],  # N(i)
                x[..., 1:, 1, :],  # CA(i)
                x[..., 1:, 2, :],  # C(i)
            ],
            dim=-2,
        )
        for x in (positions, mask)
    )  # [..., L, 4, 3]

    mask = mask.all(dim=-1).all(dim=-1)  # [..., L]

    angles = coordinates_to_dihedral(input)
    nan_tensor = torch.full_like(angles[..., :1], float("nan"))
    false_tensor = torch.zeros_like(mask[..., :1])

    angles = torch.cat([nan_tensor, angles], dim=-1)
    mask = torch.cat([false_tensor, mask], dim=-1)
    mask = mask & chain_boundary_mask & chain_break_mask

    return angles, mask


def _positions_to_psi(
    positions: torch.Tensor,
    mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
):
    chain_boundary_mask = torch.cat(
        [torch.diff(chain_index, n=1, dim=-1) == 0, torch.zeros_like(chain_index[..., :1], dtype=torch.bool)], dim=-1
    )

    chain_break_mask = torch.cat(
        [torch.diff(residue_index, n=1, dim=-1) == 1, torch.zeros_like(residue_index[..., :1], dtype=torch.bool)],
        dim=-1,
    )

    (input, mask) = (
        torch.stack(
            [
                x[..., :-1, 0, :],  # N(i)
                x[..., :-1, 1, :],  # CA(i)
                x[..., :-1, 2, :],  # C(i)
                x[..., 1:, 0, :],  # N(i+1)
            ],
            dim=-2,
        )
        for x in (positions, mask)
    )  # [..., L, 4, 3]

    mask = mask.all(dim=-1).all(dim=-1)  # [..., L]

    angles = coordinates_to_dihedral(input)
    nan_tensor = torch.full_like(angles[..., :1], float("nan"))
    false_tensor = torch.zeros_like(mask[..., :1])

    angles = torch.cat([angles, nan_tensor], dim=-1)
    mask = torch.cat([mask, false_tensor], dim=-1)
    mask = mask & chain_boundary_mask & chain_break_mask

    return angles, mask


def _positions_to_omega(
    positions: torch.Tensor,
    mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
):
    chain_boundary_mask = torch.cat(
        [torch.diff(chain_index, n=1, dim=-1) == 0, torch.zeros_like(chain_index[..., :1], dtype=torch.bool)], dim=-1
    )

    chain_break_mask = torch.cat(
        [torch.diff(residue_index, n=1, dim=-1) == 1, torch.zeros_like(residue_index[..., :1], dtype=torch.bool)],
        dim=-1,
    )

    (input, mask) = (
        torch.stack(
            [
                x[..., :-1, 1, :],  # CA(i)
                x[..., :-1, 2, :],  # C(i)
                x[..., 1:, 0, :],  # N(i+1)
                x[..., 1:, 1, :],  # CA(i+1)
            ],
            dim=-2,
        )
        for x in (positions, mask)
    )  # [..., L, 4, 3]

    mask = mask.all(dim=-1).all(dim=-1)  # [..., L]

    angles = coordinates_to_dihedral(input)
    nan_tensor = torch.full_like(angles[..., :1], float("nan"))
    false_tensor = torch.zeros_like(mask[..., :1])

    angles = torch.cat([angles, nan_tensor], dim=-1)
    mask = torch.cat([mask, false_tensor], dim=-1)
    mask = mask & chain_boundary_mask & chain_break_mask

    return angles, mask


def positions_to_backbone_dihedrals(
    positions: torch.Tensor, mask: torch.Tensor, residue_index: torch.Tensor | None = None, chain_index: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Backbone dihedral angles (phi, psi, omega) from the atom-wise coordinates.

    Parameters
    ----------
    positions: Tensor
        Shape (..., L, 37, 3) tensor of the atom-wise coordinates
    mask: BoolTensor
        Shape (..., L, 37) boolean tensor indicating which atoms are present
    residue_index: Tensor | None = None
        Optional shape (..., L) tensor specifying the index of each residue along its chain.
        If supplied this is used to mask dihedrals that cross a chain break.
    chain_index: Tensor | None = None
        Optional shape (..., L) tensor specifying the chain index of each residue.
        If supplied this is used to mask dihedrals that cross a chain boundary.

    where `NAW` is the number of atoms in the atom wide representation.

    Returns
    -------
    dihedrals: Tensor
        Shape (..., L, 3) tensor of the dihedral angles (phi, psi, omega)
    dihedrals_mask: BoolTensor
        Shape (..., L, 3) boolean tensor indicating which dihedrals are present
    """
    assert positions.ndim >= 3
    L = positions.shape[-3]
    device = positions.device

    if residue_index is None:
        residue_index = torch.arange(L).expand(*positions.shape[:-3], -1)  # [..., L]
        residue_index = residue_index.to(device)

    if chain_index is None:
        chain_index = torch.zeros_like(positions[..., :, 0, 0], dtype=torch.int64)  # [..., L]
        chain_index = chain_index.to(device)

    mask = mask.unsqueeze(-1).expand(*mask.shape,3)
    phi, phi_mask = _positions_to_phi(positions, mask, residue_index=residue_index, chain_index=chain_index)
    psi, psi_mask = _positions_to_psi(positions, mask, residue_index=residue_index, chain_index=chain_index)
    omega, omega_mask = _positions_to_omega(positions, mask, residue_index=residue_index, chain_index=chain_index)

    dihedrals = torch.stack([phi, psi, omega], dim=-1)
    dihedrals_mask = torch.stack([phi_mask, psi_mask, omega_mask], dim=-1)

    return dihedrals, dihedrals_mask


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

