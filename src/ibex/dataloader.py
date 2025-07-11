# Copyright 2025 Genentech
# Copyright 2024 Exscientia
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

import random
import pandas as pd
import torch
from tqdm import tqdm
import itertools
from math import ceil
from torch import Tensor
from typing import Iterator, List, Dict, Optional, Union
from pathlib import Path
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, BatchSampler, Sampler
from collections.abc import Iterable
from torch.utils.data.distributed import DistributedSampler

from ibex.openfold.utils.residue_constants import restype_order_with_x

# Fileids from the test and validation set that are not present in our version of the dataset, or
# where there are mismatches in the lengths of the chains.
invalid_fileids = [
    # "8aon_BC", # valid
    # "4hjj_HL", # valid
    # "7pa6_KKKKKK", # valid
    # "7ce2_ZB", # valid
    # "7r8u_HHHLLL", # valid
    # "6o89_HL", # valid
    # "5d6c_HL", # valid
    # "7u8c_HL", # test
    # "7seg_HL", # test
    # "2a9m_IM", # test
    # "4buh_AA", # test
    # "6ss5_HHHLLL" # test
    "7sgm_HL", # mismatching lengths
    "7sgm_XY", # mismatching lengths
    "7sgm_KM", # mismatching lengths
    "6w9g_HL", # mismatching lengths
    "6w9g_KM", # mismatching lengths
    "6w9g_XY", # mismatching lengths
    "5iwl_AB", # mismatching lengths
    "5iwl_BA", # mismatching lengths
    "7sen_HL", # mismatching lengths
    "6w5a_HL", # mismatching lengths
    "6bjz_HL", # mismatching lengths
    "7t74_HL", # mismatching lengths
    "6dfv_BA", # mismatching lengths TCR
    "3tyf_DC", # mismatching lengths TCR
]

class ABDataset(Dataset):
    def __init__(
        self,
        split: str,
        split_file: str,
        data_dir: str,
        limit: Optional[int] = None,
        use_vhh: bool = False,
        use_tcr: bool = False,
        rel_pos_dim: int = 16,
        edge_chain_feature: bool = False,
        use_plm_embeddings: bool = False,
        use_public_only: bool = False,
        use_private_only: bool = False,
        cluster_column: str = "cluster",
        weight_temperature: float = 1.0,
        weight_clip_quantile: float = 0.99,
        vhh_weight: float = 1.0,
        tcr_weight: float = 1.0,
        matched_weight: float = 1.0,
        conformation_node_feature: bool = False,
        use_contrastive: bool = False,
        is_matched: bool = False,
        use_boltz_only: bool = False,
        use_weights: bool = True,
    ) -> None:
        """Dataset of antibodies suitable for openfold structuremodules and loss functions.

        Args:
            path (str): root data folder
            split (str): "all", "train", "valid" or "test"
        """
        super().__init__()
        self.split = split
        self.limit = limit
        self.split_file = split_file
        self.data_dir = data_dir
        self.rel_pos_dim = rel_pos_dim
        self.edge_chain_feature = edge_chain_feature
        self.use_plm_embeddings = use_plm_embeddings
        self.conformation_node_feature = conformation_node_feature
        self.use_contrastive = use_contrastive
        self.is_matched = is_matched
        self.cluster_column = cluster_column
        self.weight_temperature = weight_temperature
        self.weight_clip_quantile = weight_clip_quantile

        self.df = pd.read_csv(split_file, index_col=0, dtype={'file_id': str})
        if not use_vhh:
            if "is_vhh" in self.df.columns:
                self.df = self.df[self.df["is_vhh"] == False]
        if not use_tcr:
            if "is_tcr" in self.df.columns:
                self.df = self.df[self.df["is_tcr"] == False]
        if split != "all":
            self.df = self.df.query(f"split=='{split}'")
        if use_public_only:
            self.df = self.df[self.df["is_internal"]==False]
        if use_private_only:
            self.df = self.df[self.df["is_internal"]]
        if 'is_boltz' in self.df.columns and use_boltz_only:
            self.df = self.df[self.df["is_boltz"]]
        if use_contrastive:
            self.df = self.df[self.df["is_matched"]==is_matched]

        self.df = self.df[~self.df.index.isin(invalid_fileids)]

        if limit is not None:
            self.df = self.df.iloc[:limit]

        if cluster_column in self.df.columns and use_weights:
            # Get basic weight that is 1/cluster size
            probs = self.df[cluster_column].map(self.df.groupby(cluster_column).size()).values
            probs = torch.tensor(probs, dtype=torch.float32)
            probs = probs / probs.sum()
            self.weights = 1 / (probs ** self.weight_temperature)
            # Cap the weight to e.g. 0.99 quantile, to avoid bad outliers
            self.weights = torch.minimum(self.weights, torch.quantile(self.weights, self.weight_clip_quantile))
            # Make the weight sum to 1
            self.weights = self.weights / self.weights.sum()
        else:
            logger.info(f"Using uniform weights for {split_file}")
            self.weights = torch.ones(len(self.df)) / len(self.df)

        # Upweight TCRs and VHHs
        if use_vhh:
            self.weights[self.df["is_vhh"]] *= vhh_weight
        if use_tcr:
            self.weights[self.df["is_tcr"]] *= tcr_weight
        if 'is_matched' in self.df.columns:
            self.weights[self.df["is_matched"]] *= matched_weight
        self.weights = self.weights / self.weights.sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        if self.use_contrastive and self.is_matched:
            datapoint_pos = self._load_single_sample(self.df.iloc[idx].name)
            datapoint_neg = self._load_single_sample(self.df.iloc[idx].matched_index)

            datapoint_pos["is_matched"] = torch.tensor(True, dtype=torch.bool)
            datapoint_neg["is_matched"] = torch.tensor(True, dtype=torch.bool)

            return datapoint_pos, datapoint_neg
            # return datapoint_pos, datapoint_neg, datapoint_anchor
        elif self.split=='train' or self.split=='all':
            datapoint = self._load_single_sample(self.df.iloc[idx].name)
            # Get a random pairing respecting the weights
            idx_other = random.choices(
                range(len(self.df)),
                weights=self.weights,
                k=1,
            )[0]
            datapoint_other = self._load_single_sample(self.df.iloc[idx_other].name)

            datapoint["is_matched"] = torch.tensor(False, dtype=torch.bool)
            datapoint_other["is_matched"] = torch.tensor(False, dtype=torch.bool)

            return datapoint, datapoint_other
        else:
            datapoint = self._load_single_sample(self.df.iloc[idx].name)
            return datapoint

    def _load_single_sample(self, file_id):
        fname = Path(self.data_dir) / f"{file_id}.pt"
        datapoint = torch.load(fname, weights_only=False)
        datapoint.update(
            self.single_and_double_from_datapoint(
                datapoint,
                self.rel_pos_dim,
                self.edge_chain_feature,
                self.use_plm_embeddings,
                self.conformation_node_feature,
            )
        )
        for key in datapoint:
            if isinstance(datapoint[key], torch.Tensor):
                datapoint[key] = datapoint[key].detach()
        return datapoint

    @staticmethod
    def single_and_double_from_datapoint(
        datapoint: dict,
        rel_pos_dim: int,
        edge_chain_feature: bool = False,
        use_plm_embeddings: bool = False,
        conformation_node_feature: bool = False,
    ):
        """
        datapoint is a dict containing:
            aatype - [n,] tensor of ints for the amino acid (including unknown)
            is_heavy - [n,] tensor of ints where 1 is heavy chain and 0 is light chain.
            residue_index - [n,] tensor of ints assinging integer to each residue

        rel_pos_dim: integer determining edge feature dimension

        edge_chain_feature: boolean to add an edge feature z_ij that encodes what chain i and j are in.

        returns:
            A dictionary containing single a tensor of size (n, 23) and pair a tensor of size (n, n, 2 * rel_pos_dim + 1 + x) where x is 3 if edge_chain_feature and 0 otherwise.
        """
        single_aa = torch.nn.functional.one_hot(datapoint["aatype"], 21)
        single_chain = torch.nn.functional.one_hot(datapoint["is_heavy"].long(), 2)
        if conformation_node_feature:
            single_conformation = torch.nn.functional.one_hot(datapoint["is_apo"].long(), 2)
            single = torch.cat((single_aa, single_chain, single_conformation), dim=-1)
        else:
            single = torch.cat((single_aa, single_chain), dim=-1)
        pair = datapoint["residue_index"]
        pair = pair[None] - pair[:, None]
        pair = pair.clamp(-rel_pos_dim, rel_pos_dim) + rel_pos_dim
        pair = torch.nn.functional.one_hot(pair, 2 * rel_pos_dim + 1)
        if edge_chain_feature:
            is_heavy = datapoint["is_heavy"]
            is_heavy = 2 * is_heavy.outer(is_heavy) + (
                (1 - is_heavy).outer(1 - is_heavy)
            )
            is_heavy = torch.nn.functional.one_hot(is_heavy.long())
            pair = torch.cat((is_heavy, pair), dim=-1)
        if use_plm_embeddings:
            {"single": single.float(), "pair": pair.float(), "plm_embedding": datapoint["plm_embedding"]}
        return {"single": single.float(), "pair": pair.float()}

def pad_square_tensors(tensors: list[torch.tensor]) -> torch.tensor:
    """Pads a list of tensors in the first two dimensions.

    Args:
        tensors (list[torch.tensor]): Input tensor are of shape (n_1, n_1, ...), (n_2, n_2, ...). where shape matches in the ... dimensions

    Returns:
        torch.tensor: A tensor of size (len(tensor), max(n_1,...), max(n_1,...), ...)
    """
    max_len = max(map(len, tensors))
    output = torch.zeros((len(tensors), max_len, max_len, *tensors[0].shape[2:]))
    for i, tensor in enumerate(tensors):
        output[i, : tensor.size(0), : tensor.size(1)] = tensor
    return output


pad_first_dim_keys = [
    "atom14_gt_positions",
    "atom14_alt_gt_positions",
    "atom14_atom_is_ambiguous",
    "atom14_gt_exists",
    "atom14_alt_gt_exists",
    "atom14_atom_exists",
    "single",
    "plm_embedding",
    "seq_mask",
    "aatype",
    "backbone_rigid_tensor",
    "backbone_rigid_mask",
    "rigidgroups_gt_frames",
    "rigidgroups_alt_gt_frames",
    "rigidgroups_gt_exists",
    "cdr_mask",
    "chi_mask",
    "chi_angles_sin_cos",
    "residue_index",
    "residx_atom14_to_atom37",
    "region_numeric",
]

pad_first_two_dim_keys = ["pair"]


def string_to_input(heavy: str, light: str, apo: bool = False, conformation_aware: bool = False, embedding=None, device: str = "cpu") -> dict:
    """Generates an input formatted for an Ibex model from heavy and light chain
    strings.

    Args:
        heavy (str): heavy chain
        light (str): light chain
        apo (bool): whether the structure is apo or holo (optional)

    Returns:
        dict: A dictionary containing
            aatype: an (n,) tensor of integers encoding the amino acid string
            is_heavy: an (n,) tensor where is_heavy[i] = 1 means residue i is heavy and
                is_heavy[i] = 0 means residue i is light
            residue_index: an (n,) tensor with indices for each residue. There is a gap
                of 500 between the last heavy residue and the first light residue
            single: a (1, n, 23) tensor of node features
            pair: a (1, n, n, 132) tensor of edge features
    """
    aatype = []
    is_heavy = []
    for character in heavy:
        is_heavy.append(1)
        aatype.append(restype_order_with_x[character])
    if light is not None:
        for character in light:
            is_heavy.append(0)
            aatype.append(restype_order_with_x[character])
    is_heavy = torch.tensor(is_heavy)
    aatype = torch.tensor(aatype)
    if light is None:
        residue_index = torch.arange(len(heavy))
    else:
        residue_index = torch.cat(
            (torch.arange(len(heavy)), torch.arange(len(light)) + 500)
        )

    model_input = {
        "is_heavy": is_heavy,
        "aatype": aatype,
        "residue_index": residue_index,
    }

    if apo and conformation_aware:
        model_input["is_apo"] = torch.ones(len(heavy + (light if light is not None else '')),dtype=torch.int64)
    elif conformation_aware:
        model_input["is_apo"] = torch.zeros(len(heavy + (light if light is not None else '')),dtype=torch.int64)
    if embedding is not None:
        model_input["plm_embedding"] = embedding
    model_input.update(
        ABDataset.single_and_double_from_datapoint(
            model_input, 64, edge_chain_feature=True, conformation_node_feature=conformation_aware
        )
    )
    if "plm_embedding" in model_input:
        model_input["plm_embedding"] = model_input["plm_embedding"].unsqueeze(0)
    model_input["single"] = model_input["single"].unsqueeze(0)
    model_input["pair"] = model_input["pair"].unsqueeze(0)
    model_input = {k: v.to(device) for k, v in model_input.items()}
    return model_input


def collate_fn(batch: dict):
    """A collate function so the ABDataset can be used in a torch dataloader.

    Args:
        batch (dict): A list of datapoints from ABDataset

    Returns:
        dict: A dictionary where the keys are the same as batch but map to a batched tensor where the batch is on the leading dimension.
    """
    # Flatten the batch structure (needed for contrastive examples, where some of the batch elements are tuples of objects)
    flattened_batch = []
    for item in batch:
        if isinstance(item, tuple):
            flattened_batch.extend(item)
        else:
            flattened_batch.append(item)
    batch = {key: [d[key] for d in flattened_batch] for key in flattened_batch[0]}
    output = {}
    for key in batch:
        if key in pad_first_dim_keys:
            output[key] = pad_sequence(batch[key], batch_first=True)
        elif key in pad_first_two_dim_keys:
            output[key] = pad_square_tensors(batch[key])
        elif key == "resolution":
            output[key] = torch.Tensor(batch["resolution"])
        elif key == "is_matched":
            output[key] = torch.stack(batch[key])
    return output


class SequenceDataset(Dataset):
    def __init__(self, fv_heavy_list, fv_light_list, apo_list=None, plm_model=None, batch_size=2, num_workers=0):
        self.fv_heavy_list = fv_heavy_list
        self.fv_light_list = fv_light_list
        self.apo_list = apo_list
        if plm_model is not None:
            with torch.no_grad():
                self.fv_heavy_embedding = plm_model.embed_dataset(
                    fv_heavy_list,
                    batch_size=batch_size,
                    max_len=max([len(x) for x in fv_heavy_list]),
                    full_embeddings=True,
                    full_precision=False,
                    pooling_type="mean",
                    num_workers=num_workers,
                    sql=False
                )
                self.fv_light_embedding = plm_model.embed_dataset(
                    [light for light in fv_light_list if light is not None and light!=''],
                    batch_size=batch_size,
                    max_len=max([len(x) for x in fv_light_list]),
                    full_embeddings=True,
                    full_precision=False,
                    pooling_type="mean",
                    num_workers=num_workers,
                    sql=False
                ) if fv_light_list is not None else None
        else:
            self.fv_heavy_embedding = None
            self.fv_light_embedding = None

    def __len__(self):
        return len(self.fv_heavy_list)

    def __getitem__(self, idx):
        heavy = self.fv_heavy_list[idx]
        light = self.fv_light_list[idx] if self.fv_light_list is not None else None
        if self.fv_heavy_embedding is not None and self.fv_light_embedding is not None:
            if light is None:
                embedding = self.fv_heavy_embedding[heavy]
            else:
                embedding = torch.concat([self.fv_heavy_embedding[heavy], self.fv_light_embedding[light]])
        else:
            embedding = None
        if self.apo_list is not None:
            apo = self.apo_list[idx]
            return string_to_input(heavy, light, apo=apo, conformation_aware=True, embedding=embedding)
        return string_to_input(heavy, light, embedding=embedding)

def sequence_collate_fn(batch):
    # Separating each component of the batch
    batch_dict = {key: [d[key] if key not in ["single","pair","plm_embedding"] else d[key].squeeze(0) for d in batch] for key in batch[0]}
    # Prepare output dictionary
    output = {}
    # Pad the "single" features
    output["single"] = pad_sequence(batch_dict["single"], batch_first=True)
    # Pad the "pair" features
    output["pair"] = pad_square_tensors(batch_dict["pair"])
    # Copy other keys directly
    for key in ["aatype", "residue_index", "is_heavy"]:
        output[key] = pad_sequence(batch_dict[key], batch_first=True)
    if "is_apo" in batch_dict:
        output["is_apo"] = pad_sequence(batch_dict["is_apo"], batch_first=True)
    if "plm_embedding" in batch_dict:
        output["plm_embedding"] = pad_sequence(batch_dict["plm_embedding"], batch_first=True)
    # Create mask based on the lengths of "aatype"
    mask = [torch.ones(len(aatype)) for aatype in batch_dict["aatype"]]
    output["mask"] = pad_sequence(mask, batch_first=True)
    return output

if __name__ == "__main__":
    from torch.utils.data import DataLoader


    dataset = ABDataset("test", split_file="/data/dreyerf1/ibex/split.csv", data_dir="/data/dreyerf1/ibex/structures", edge_chain_feature=True)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    for batch in dataloader:
        break
    for key in batch:
        print(key, batch[key].shape)
