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

import lightning.pytorch as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LinearLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import ConcatDataset
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from ibex.dataloader import ABDataset, collate_fn, string_to_input, sequence_collate_fn
from ibex.loss import IbexLoss
from ibex.utils import output_to_pdb, add_atom37_to_output, output_to_protein
from ibex.openfold.model import StructureModule

class IbexDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_dir_sabdab: str,
        split_file_sabdab: str,
        data_dir_predicted: str,
        split_file_predicted: str,
        fraction_predicted: float,
        data_dir_ig: str,
        split_file_ig: str,
        fraction_ig: float,
        fraction_matched: float,
        use_vhh: bool = False,
        use_tcr: bool = False,
        public_only_data: bool = False,
        rel_pos_dim: int = 64,
        edge_chain_feature: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_plm_embeddings: bool = False,
        use_weighted_sampler: bool = False,
        cluster_column: str = "cluster",
        weight_temperature: float = 1.0,
        weight_clip_quantile: float = 0.99,
        vhh_weight: float = 1.0,
        tcr_weight: float = 1.0,
        matched_weight: float = 1.0,
        conformation_node_feature: bool = False,
        use_contrastive: bool = False,
        use_boltz_only: bool = False,
    ):
        super().__init__()
        self.data_dir_sabdab = data_dir_sabdab
        self.data_dir_predicted = data_dir_predicted
        self.data_dir_ig = data_dir_ig
        self.split_file_sabdab = split_file_sabdab
        self.split_file_predicted = split_file_predicted
        self.split_file_ig = split_file_ig
        self.fraction_predicted = fraction_predicted
        self.fraction_ig = fraction_ig
        self.fraction_matched = fraction_matched if use_contrastive else 0.0
        self.batch_size = batch_size
        self.use_vhh = use_vhh
        self.use_tcr = use_tcr
        self.public_only_data = public_only_data
        self.rel_pos_dim = rel_pos_dim
        self.edge_chain_feature = edge_chain_feature
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_plm_embeddings = use_plm_embeddings
        self.use_weighted_sampler = use_weighted_sampler
        self.cluster_column = cluster_column
        self.weight_temperature = weight_temperature
        self.weight_clip_quantile = weight_clip_quantile
        self.vhh_weight = vhh_weight
        self.tcr_weight = tcr_weight
        self.matched_weight = matched_weight
        self.conformation_node_feature = conformation_node_feature
        self.use_contrastive = use_contrastive
        self.use_boltz_only = use_boltz_only

    def setup(self, stage: str):
        sabdab_dataset = ABDataset(
            "train",
            split_file=self.split_file_sabdab,
            data_dir=self.data_dir_sabdab,
            use_vhh=self.use_vhh,
            use_tcr=self.use_tcr,
            rel_pos_dim=self.rel_pos_dim,
            edge_chain_feature=self.edge_chain_feature,
            use_plm_embeddings=self.use_plm_embeddings,
            use_public_only=self.public_only_data,
            cluster_column=self.cluster_column,
            weight_temperature=self.weight_temperature,
            weight_clip_quantile=self.weight_clip_quantile,
            vhh_weight=self.vhh_weight,
            tcr_weight=self.tcr_weight,
            matched_weight=self.matched_weight,
            conformation_node_feature=self.conformation_node_feature,
            use_contrastive=self.use_contrastive,
            is_matched=False,
            use_weights=self.use_weighted_sampler,
        )
        sabdab_dataset_matched = ABDataset(
            "train",
            split_file=self.split_file_sabdab,
            data_dir=self.data_dir_sabdab,
            use_vhh=self.use_vhh,
            use_tcr=self.use_tcr,
            rel_pos_dim=self.rel_pos_dim,
            edge_chain_feature=self.edge_chain_feature,
            use_plm_embeddings=self.use_plm_embeddings,
            use_public_only=self.public_only_data,
            cluster_column=self.cluster_column,
            weight_temperature=self.weight_temperature,
            weight_clip_quantile=self.weight_clip_quantile,
            vhh_weight=self.vhh_weight,
            tcr_weight=self.tcr_weight,
            conformation_node_feature=self.conformation_node_feature,
            use_contrastive=self.use_contrastive,
            is_matched=True,
            use_weights=self.use_weighted_sampler,
        ) if self.use_contrastive else None
        predicted_dataset = ABDataset(
            "all",
            split_file=self.split_file_predicted,
            data_dir=self.data_dir_predicted,
            use_vhh=False,
            use_tcr=False,
            rel_pos_dim=self.rel_pos_dim,
            edge_chain_feature=self.edge_chain_feature,
            use_plm_embeddings=self.use_plm_embeddings,
            use_public_only=self.public_only_data,
            cluster_column=self.cluster_column,
            weight_temperature=self.weight_temperature,
            weight_clip_quantile=self.weight_clip_quantile,
            vhh_weight=self.vhh_weight,
            tcr_weight=self.tcr_weight,
            conformation_node_feature=self.conformation_node_feature,
            use_boltz_only=self.use_boltz_only,
            use_weights=self.use_weighted_sampler,
        ) if self.fraction_predicted > 0.0 else None
        ig_dataset = ABDataset(
            "all",
            split_file=self.split_file_ig,
            data_dir=self.data_dir_ig,
            use_vhh=True,
            use_tcr=False,
            rel_pos_dim=self.rel_pos_dim,
            edge_chain_feature=self.edge_chain_feature,
            use_plm_embeddings=self.use_plm_embeddings,
            use_public_only=self.public_only_data,
            cluster_column=self.cluster_column,
            weight_temperature=self.weight_temperature,
            weight_clip_quantile=self.weight_clip_quantile,
            vhh_weight=self.vhh_weight,
            tcr_weight=self.tcr_weight,
            conformation_node_feature=self.conformation_node_feature,
            use_weights=self.use_weighted_sampler,
        ) if self.fraction_ig > 0.0 else None
        self.train_dataset = ConcatDataset([d for d in [sabdab_dataset, predicted_dataset, ig_dataset, sabdab_dataset_matched] if d is not None])
        self.len_train_dataset = len(sabdab_dataset)
        sabdab_fraction = 1.0 - self.fraction_predicted - self.fraction_ig - self.fraction_matched
        self.train_dataset_weights = torch.cat(
                [d.weights * frac for d, frac in [(sabdab_dataset, sabdab_fraction), (predicted_dataset, self.fraction_predicted), (ig_dataset, self.fraction_ig), (sabdab_dataset_matched, self.fraction_matched)] if d is not None]
            )
        self.train_dataset_weights = self.train_dataset_weights / self.train_dataset_weights.sum()
        self.valid_dataset = ABDataset(
            "valid",
            split_file=self.split_file_sabdab,
            data_dir=self.data_dir_sabdab,
            use_vhh=False,
            use_tcr=False,
            rel_pos_dim=self.rel_pos_dim,
            edge_chain_feature=self.edge_chain_feature,
            use_plm_embeddings=self.use_plm_embeddings,
            use_public_only=self.public_only_data,
            conformation_node_feature=self.conformation_node_feature,
        )
        self.test_dataset = ABDataset(
            "test",
            split_file=self.split_file_sabdab,
            data_dir=self.data_dir_sabdab,
            use_vhh=False,
            use_tcr=False,
            rel_pos_dim=self.rel_pos_dim,
            edge_chain_feature=self.edge_chain_feature,
            use_plm_embeddings=self.use_plm_embeddings,
            use_public_only=self.public_only_data,
            conformation_node_feature=self.conformation_node_feature,
        )
        if stage == "test":
            self.test_public_dataset = ABDataset(
                "test",
                split_file=self.split_file_sabdab,
                data_dir=self.data_dir_sabdab,
                use_vhh=False,
                use_tcr=False,
                rel_pos_dim=self.rel_pos_dim,
                edge_chain_feature=self.edge_chain_feature,
                use_plm_embeddings=self.use_plm_embeddings,
                use_public_only=True,
                conformation_node_feature=self.conformation_node_feature,
            )
            self.test_public_vhh_dataset = ABDataset(
                "test_vhh",
                split_file=self.split_file_sabdab,
                data_dir=self.data_dir_sabdab,
                use_vhh=True,
                use_tcr=False,
                rel_pos_dim=self.rel_pos_dim,
                edge_chain_feature=self.edge_chain_feature,
                use_plm_embeddings=self.use_plm_embeddings,
                use_public_only=True,
                conformation_node_feature=self.conformation_node_feature,
            )
            self.test_public_tcr_dataset = ABDataset(
                "test_tcr",
                split_file=self.split_file_sabdab,
                data_dir=self.data_dir_sabdab,
                use_vhh=False,
                use_tcr=True,
                rel_pos_dim=self.rel_pos_dim,
                edge_chain_feature=self.edge_chain_feature,
                use_plm_embeddings=self.use_plm_embeddings,
                use_public_only=True,
                conformation_node_feature=self.conformation_node_feature,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            sampler=WeightedRandomSampler(self.train_dataset_weights, num_samples=self.len_train_dataset // 2, replacement=self.use_weighted_sampler),
            batch_size=self.batch_size // 2, # Each element in our dataset actually contains 2 samples
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_public_dataloader(self):
        return DataLoader(
            self.test_public_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_public_vhh_dataloader(self):
        return DataLoader(
            self.test_public_vhh_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_public_tcr_dataloader(self):
        return DataLoader(
            self.test_public_tcr_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

class Ibex(pl.LightningModule):
    def __init__(self, model_config, loss_config, optim_config, conformation_aware=False, use_plm=False, ensemble=False, models=None, stage=None, init_plm=False):
        super().__init__()
        model_config["use_plddt"] = loss_config.plddt.weight > 0
        self.save_hyperparameters()
        self.model_config = model_config
        self.loss_config = loss_config
        self.optim_config = optim_config
        self.loss = IbexLoss(loss_config)
        self.finetune = False
        self.contrastive = False
        self.ensemble = ensemble
        self.conformation_aware = conformation_aware
        self.stage = stage

        if self.ensemble:
            if models is None:
                raise ValueError("Models must be provided for ensemble mode.")
            # Use the provided models to create the EnsembleStructureModule
            self.model = EnsembleStructureModule(models)
        else:
            self.model = StructureModule(**model_config)

        if use_plm:
            # # self.plm_model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True)
            # self.plm_model = ESMC.from_pretrained("esmc_300m")
            # self.plm_model.eval()
            # for param in self.plm_model.parameters():
            #     param.requires_grad = False
            self.plm_model = True
            if init_plm:
                self.set_plm()
            # self.plm_model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True)
        else:
            self.plm_model = None

    def set_plm(self):
        if self.plm_model is not None:
            self.plm_model = ESMC.from_pretrained("esmc_300m")
            self.plm_model.eval()
            for param in self.plm_model.parameters():
                param.requires_grad = False

    @classmethod
    def load_from_ensemble_checkpoint(cls, checkpoint_path, map_location=None):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state_dicts = checkpoint['state_dicts']
        model_config = checkpoint['model_config']
        loss_config = checkpoint['loss_config']
        optim_config = checkpoint['optim_config']
        if 'conformation_aware' in checkpoint:
            conformation_aware = checkpoint['conformation_aware']
        else:
            conformation_aware = False
        if 'language' in checkpoint:
            init_plm = checkpoint['language']
        else:
            init_plm = False
        # Initialize the models from state dicts
        models = [StructureModule(**model_config) for _ in state_dicts]
        for model, state_dict in zip(models, state_dicts):
            model.load_state_dict(state_dict)

        return cls(model_config, loss_config, optim_config, conformation_aware=conformation_aware, ensemble=True, models=models, use_plm=init_plm, init_plm=init_plm)

    def training_step(self, batch, batch_idx):
        stage=f"_{self.stage}" if self.stage else ""
        loss = self._step(batch, "train"+stage)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=self.optim_config.max_grad_norm
        )
        self.log(f"monitor{stage}/grad_norm", grad_norm)
        # Log custom epoch and learning rate
        if stage:
            self.log(
                f"monitor{stage}/epoch",
                self.current_epoch
            )
            self.log(
                f"monitor{stage}/learning_rate",
                self.trainer.optimizers[0].param_groups[0]['lr']
            )
        return loss

    def validation_step(self, batch, batch_idx):
        stage=f"_{self.stage}" if self.stage else ""
        self._step(batch, "valid"+stage)
        self.log(
            f"monitor{stage}/finetune",
            float(self.finetune),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _step(self, batch, split):
        if self.plm_model is not None:
            output = self.model(
                {
                    "single": batch["single"],
                    "pair": batch["pair"],
                    "plm_embedding": batch["plm_embedding"],
                },
                batch["aatype"],
                batch["seq_mask"],
            )
        else:
            output = self.model(
                {
                    "single": batch["single"],
                    "pair": batch["pair"],
                },
                batch["aatype"],
                batch["seq_mask"],
            )
        loss, loss_dict = self.loss(output, batch, self.finetune, (self.contrastive and "train" in split))
        for loss_name in loss_dict:
            self.log(
                f"{split}/{loss_name}",
                loss_dict[loss_name],
                prog_bar=loss_name == "loss",
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
        return loss

    def configure_optimizers(self):
        if self.optim_config.optimizer == "AdamW":
            if self.stage=='first_stage':
                optimizer = AdamW(
                    self.parameters(),
                    lr=self.optim_config.lr,
                    betas=(0.9, 0.99),
                    eps=1e-6,
                )
                scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=self.optim_config.linear_iters)
            else:
                optimizer = AdamW(
                    self.parameters(),
                    lr=self.optim_config.lr,
                    betas=(0.9, 0.99),
                    eps=1e-6,
                )
                scheduler = LambdaLR(
                    optimizer, lambda epoch: max(self.optim_config.lambda_lr ** (epoch-self.optim_config.lambda_iters), self.optim_config.lambda_min_factor)
                    if epoch >= self.optim_config.lambda_iters else 1
                )

            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            }
            return [optimizer], [lr_scheduler]
        else:
            raise ValueError(
                "Expected AdamW as optimizer. Instead got"
                f" {self.optim_config.optimizer=}."
            )

    def predict(self, fv_heavy, fv_light, device, ensemble=False, pdb_string=True, apo=False):
        self.model.eval()
        self.to(device)
        if self.plm_model is not None:
            self.plm_model.to(device)
            if fv_light is None or fv_light=="":
                # for nanobodies, embed the heavy chain
                prot_h = ESMProtein(sequence=fv_heavy)
                prot_tensor_h = self.plm_model.encode(prot_h)
                logits_output_h = self.plm_model.logits(
                    prot_tensor_h, LogitsConfig(sequence=True, return_embeddings=True)
                )
                embedding = logits_output_h.embeddings[0,1:-1]
                # tokenized = self.plm_model.tokenizer([fv_heavy], padding=True, return_tensors='pt')
                # output = self.plm_model(**tokenized)
                # embedding = output.last_hidden_state[0][1:-1]
            else:
                # for antibodies, embed both chains
                prot_h = ESMProtein(sequence=fv_heavy)
                prot_tensor_h = self.plm_model.encode(prot_h)
                logits_output_h = self.plm_model.logits(
                    prot_tensor_h, LogitsConfig(sequence=True, return_embeddings=True)
                )
                embed_vh = logits_output_h.embeddings[0,1:-1]

                prot_l = ESMProtein(sequence=fv_light)
                prot_tensor_l = self.plm_model.encode(prot_l)
                logits_output_l = self.plm_model.logits(
                    prot_tensor_l, LogitsConfig(sequence=True, return_embeddings=True)
                )
                embed_vl = logits_output_l.embeddings[0,1:-1]
                # tokenized = self.plm_model.tokenizer([fv_heavy,fv_light], padding=True, return_tensors='pt')
                # output = self.plm_model(**tokenized)
                # embed_vh=output.last_hidden_state[0][1:len(fv_heavy)+1]
                # embed_vl=output.last_hidden_state[1][1:len(fv_light)+1]
                embedding = torch.concat([embed_vh, embed_vl])
        else:
            # if PLM is not used, set embedding to None and use one-hot encoding instead
            embedding = None
        ab_input = string_to_input(heavy=fv_heavy, light=fv_light, apo=apo, conformation_aware=self.conformation_aware, embedding=embedding, device=device)
        ab_input_batch = {
            key: (value.unsqueeze(0) if key not in ["single", "pair", "plm_embedding"] else value)
            for key, value in ab_input.items()
        }
        # Forward pass with model
        result = self.model(ab_input_batch, ab_input_batch["aatype"], return_all=ensemble)
        if ensemble:
            predictions = []
            for i, output in enumerate(result):
                result[i] = add_atom37_to_output(output, ab_input_batch["aatype"])
                if pdb_string:
                    predictions.append(output_to_pdb(result[i], ab_input))
                else:
                    predictions.append(output_to_protein(result[i], ab_input))
            return predictions

        result = add_atom37_to_output(result, ab_input["aatype"].to(device))
        # collate the results
        if pdb_string:
            return output_to_pdb(result, ab_input)
        return output_to_protein(result, ab_input)


    def predict_batch(self, fv_heavy_batch, fv_light_batch, device, ensemble=False, pdb_string=True, apo_list=None, num_workers=0):
        self.model.eval()
        self.to(device)

        # if self.plm_model is not None:
        #     fv_heavy_embedding = self.plm_model.embed_dataset(
        #         fv_heavy_batch,
        #         batch_size=batch_size,
        #         max_len=max([len(x) for x in fv_heavy_batch]),
        #         full_embeddings=True,
        #         full_precision=False,
        #         pooling_type="mean",
        #         num_workers=num_workers,
        #         sql=False
        #     )
        #     fv_light_embedding = self.plm_model.embed_dataset(
        #         [light for light in fv_light_batch if light is not None and light!=''],
        #         batch_size=batch_size,
        #         max_len=max([len(x) for x in fv_light_batch]),
        #         full_embeddings=True,
        #         full_precision=False,
        #         pooling_type="mean",
        #         num_workers=num_workers,
        #         sql=False
        #     ) if fv_light_batch is not None else None
        # else:
        #     fv_heavy_embedding = None
        #     fv_light_embedding = None

        batch = []
        for i, fv_heavy in enumerate(fv_heavy_batch):
            fv_light = fv_light_batch[i] if fv_light_batch is not None else None
            apo = apo_list[i] if apo_list is not None else False
            # if fv_light_embedding is None and fv_heavy_embedding is not None:
            #     embedding = fv_heavy_embedding[fv_heavy][1:len(fv_heavy)+1]
            # elif fv_light_embedding is not None:
            #     embedding = torch.concat([
            #         fv_heavy_embedding[fv_heavy][1:len(fv_heavy)+1],
            #         fv_light_embedding[fv_light][1:len(fv_light)+1]
            #     ])
            # else:
            #     embedding = None
            if self.plm_model is not None:
                prot_h = ESMProtein(sequence=fv_heavy)
                prot_tensor_h = self.plm_model.encode(prot_h)
                logits_output_h = self.plm_model.logits(
                    prot_tensor_h, LogitsConfig(sequence=True, return_embeddings=True)
                )
                embed_vh = logits_output_h.embeddings[0,1:-1]
                if fv_light is None or fv_light=="":
                    # for nanobodies, embed the heavy chain
                    embedding = embed_vh
                else:
                    # for antibodies, embed both chains
                    prot_l = ESMProtein(sequence=fv_light)
                    prot_tensor_l = self.plm_model.encode(prot_l)
                    logits_output_l = self.plm_model.logits(
                        prot_tensor_l, LogitsConfig(sequence=True, return_embeddings=True)
                    )
                    embed_vl = logits_output_l.embeddings[0,1:-1]
                    embedding = torch.concat([embed_vh, embed_vl])
            else:
                embedding = None
            batch.append(string_to_input(heavy=fv_heavy, light=fv_light, apo=apo, conformation_aware=self.conformation_aware, embedding=embedding, device=device))

        ab_input_batch = sequence_collate_fn(batch)
        # Move inputs to the device
        ab_input_batch = {key: value.to(device) for key, value in ab_input_batch.items()}

        # Forward pass with mask
        results = self.model(ab_input_batch, ab_input_batch["aatype"], mask=ab_input_batch["mask"], return_all=ensemble)

        predictions = []
        batch_size = ab_input_batch["aatype"].size(0)
        if ensemble:
            for i in range(batch_size):
                ensemble_preds = []
                for result in results:
                    masked_result = {
                        "positions": result["positions"][-1,i][ab_input_batch["mask"][i]==1].unsqueeze(0).unsqueeze(0),
                    }
                    if "plddt" in result:
                        masked_result["plddt"]=result["plddt"][i][ab_input_batch["mask"][i]==1].unsqueeze(0)
                    masked_input = {
                        "aatype": ab_input_batch["aatype"][i][ab_input_batch["mask"][i]==1],
                        "is_heavy": ab_input_batch["is_heavy"][i][ab_input_batch["mask"][i]==1],
                    }
                    # Add atom37 coordinates to the output
                    masked_result = add_atom37_to_output(masked_result, masked_input["aatype"].unsqueeze(0))
                    if pdb_string:
                        ensemble_preds.append(output_to_pdb(masked_result, masked_input))
                    else:
                        ensemble_preds.append(output_to_protein(masked_result, masked_input))
                predictions.append(ensemble_preds)
            return predictions

        # Iterate over each item in the batch
        for i in range(batch_size):
            masked_result = {
                "positions": results["positions"][-1,i][ab_input_batch["mask"][i]==1].unsqueeze(0).unsqueeze(0),
            }
            if "plddt" in results:
                masked_result["plddt"]=results["plddt"][i][ab_input_batch["mask"][i]==1].unsqueeze(0)
            masked_input = {
                "aatype": ab_input_batch["aatype"][i][ab_input_batch["mask"][i]==1],
                "is_heavy": ab_input_batch["is_heavy"][i][ab_input_batch["mask"][i]==1],
            }

            # Add atom37 coordinates to the output
            masked_result = add_atom37_to_output(masked_result, masked_input["aatype"].unsqueeze(0))
            if pdb_string:
                # Generate a PDB string for the single result
                predictions.append(output_to_pdb(masked_result, masked_input))
            else:
                # Generate a Protein object for the single result
                predictions.append(output_to_protein(masked_result, masked_input))

        return predictions


class EnsembleStructureModule(torch.nn.Module):
    def __init__(self, models):
        super(EnsembleStructureModule, self).__init__()
        if not all(isinstance(model, StructureModule) for model in models):
            raise ValueError("All models must be instances of StructureModule")
        self.models = torch.nn.ModuleList(models)

    def forward(self, inputs, aatype, mask=None, return_all=False, plddt_select=False):
        outputs = []
        for model in self.models:
            output = model(inputs, aatype, mask)
            outputs.append(output)

        if return_all:
            # if all outputs are requested, no need for alignment, just return everything
            return outputs

        if aatype.shape[0] == 1:
            # if batch size is one, then we can use the original ABB2 implementation
            if plddt_select:
                from ibex.utils import compute_plddt
                plddts = torch.stack([compute_plddt(output["plddt"]).squeeze() for output in outputs])
                plddts = torch.stack([x for x in plddts]) # [E, N, 3]
                closest_index = torch.argmax(torch.quantile(plddts, 0.1, dim=1))
                print(torch.quantile(plddts, 0.1, dim=1), closest_index)
                return outputs[closest_index]
            else:
                # Stack positions along a new axis for batch processing
                positions = [output['positions'][-1].squeeze() for output in outputs]
                traces = torch.stack([x[:,0] for x in positions]) # [E, N, 3]
                # find the rotation and translation that aligns the traces
                R,t = find_alignment_transform(traces)
                aligned_traces = (traces-t) @ R
                # compute rmsd to the mean and return the prediction closest to the mean
                rmsd_values = (aligned_traces - aligned_traces.mean(0)).square().sum(-1).mean(-1)
                closest_index = torch.argmin(rmsd_values)
                return outputs[closest_index]

        # for batch size > 1 we need to perform a more sophisticated batching operation
        # and keep track of the mask
        positions = [output['positions'][-1] for output in outputs]
        traces = torch.stack([x[:,:,0] for x in positions]) # [E, B, N, 3]
        # Permute dimensions to get [B, E, N, 3]
        traces = traces.permute(1, 0, 2, 3)  # [B, E, N, 3]

        if mask is not None:
            # Expand the mask to [B, E, N, 3] for element-wise operations
            mask = mask.unsqueeze(1).unsqueeze(-1).expand(-1, traces.size(1), -1, traces.size(-1))
        R,t = find_alignment_transform_batch(traces, mask)
        aligned_traces = (traces-t) @ R

        if mask is not None:
            # Compute the mean of aligned traces along the ensemble dimension, considering the mask
            masked_aligned_traces = aligned_traces * mask  # Mask application
            mask_sum = mask.sum(1)
            mask_sum[mask_sum == 0] = 1
            mean_aligned_traces = masked_aligned_traces.sum(1) / mask_sum  # [B, N, 3]
            # Compute RMSD, taking the mask into account  -> [B, E, N]
            rmsd_values = ((aligned_traces - mean_aligned_traces.unsqueeze(1)) * mask).square().sum(-1)
             # Normalize by the number of valid elements per sequence
            rmsd_values = (rmsd_values * mask[:,:,:,0]).sum(-1) / mask[:,:,:,0].sum(-1)
        else:
            # rmsd_values = (aligned_traces - aligned_traces.mean(0)).square().sum(-1).mean(-1)
            rmsd_values = (aligned_traces - aligned_traces.mean(1).unsqueeze(1)).square().sum(-1).mean(-1)

        # Find the prediction with the minimum RMSD
        closest_index = torch.argmin(rmsd_values, dim=-1)
        result = outputs[-1]
        # now iterate over the batch and select the best prediction for each sequence
        for ibatch in range(len(closest_index)):
            for k in result:
                if k in ['single', 'plddt']:
                    result[k][ibatch] = outputs[closest_index[ibatch]][k][ibatch]
                else:
                    result[k][:, ibatch] = outputs[closest_index[ibatch]][k][:, ibatch]
        return result


def find_alignment_transform(traces):
    centers = traces.mean(-2, keepdim=True)
    traces = traces - centers

    p1, p2 = traces[0], traces[1:]
    C = torch.einsum("i j k, j l -> i k l", p2, p1)
    V, _, W = torch.linalg.svd(C)
    U = torch.matmul(V, W)
    U = torch.matmul(torch.stack([torch.ones(len(p2), device=U.device),torch.ones(len(p2), device=U.device),torch.linalg.det(U)], dim=1)[:,:,None] * V,  W)

    return torch.cat([torch.eye(3, device=U.device)[None], U]), centers


def find_alignment_transform_batch(traces, mask=None):
    # traces: [B, E, N, 3], mask: [B, E, N, 3]
    if mask is not None:
        centers = (traces * mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True)
    else:
        centers = traces.mean(dim=-2, keepdim=True)

    traces = traces - centers

    p1 = traces[:, 0, :, :]  # [B, N, 3]
    p2 = traces[:, 1:, :, :]  # [B, E-1, N, 3]

    if mask is not None:
        C = torch.einsum("...ni,...nj->...ij", p2 * mask[:, 1:], p1.unsqueeze(1) * mask[:, :1])
    else:
        C = torch.einsum("...ni,...nj->...ij", p2, p1.unsqueeze(1))

    V, _, W = torch.linalg.svd(C)

    # Compute the rotation matrix U
    U = torch.matmul(V, W)

    # Ensure U is a proper rotation matrix by checking its determinant
    det_U = torch.linalg.det(U)
    V[..., -1] *= torch.sign(det_U).unsqueeze(-1)

    U = torch.matmul(V, W)

    # Prepare identity matrices for the reference trace
    identity_matrices = torch.eye(3, device=U.device).expand(traces.size(0), 1, 3, 3)  # [B, 1, 3, 3]

    # Concatenate identity matrix for reference with U for ensemble members
    all_transforms = torch.cat([identity_matrices, U], dim=1)  # [B, E, 3, 3]

    return all_transforms, centers
