"""ChemBERTa model for CYP prediction."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models.readouts import build_readout


class SmilesDataset(Dataset):
    def __init__(self, frame, targets):
        self.smiles = frame.SMILES.tolist()
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles[i], self.y[i]


class ChemBERTaModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
        self.encoder = AutoModel.from_pretrained(cfg.checkpoint, add_pooling_layer=False)
        if cfg.max_length > self.encoder.config.max_position_embeddings:
            raise ValueError(
                f'model.max_length ({cfg.max_length}) exceeds checkpoint limit '
                f'({self.encoder.config.max_position_embeddings})'
            )
        self.encoder.requires_grad_(not cfg.freeze_encoder)
        self.encoder.to(device)
        self.readout = build_readout(readout, self.encoder.config.hidden_size, n_targets).to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def loader(self, frame, targets, batch_size, shuffle=False):
        def collate(items):
            smiles, labels = zip(*items)
            tokens = self.tokenizer(
                list(smiles),
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors='pt',
            )
            return {
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'y': torch.tensor(np.asarray(labels), dtype=torch.float32),
            }

        return DataLoader(SmilesDataset(frame, targets), batch_size=batch_size,
                          shuffle=shuffle, collate_fn=collate, num_workers=0)

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({'params': list(self.encoder.parameters()), 'lr': self.cfg.encoder_lr})
        return groups

    def forward(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        if self.cfg.freeze_encoder:
            with torch.no_grad():
                hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.readout(pooled)
