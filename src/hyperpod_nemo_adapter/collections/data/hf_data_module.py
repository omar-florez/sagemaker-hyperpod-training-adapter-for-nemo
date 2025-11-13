# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import os
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import default_data_collator

from hyperpod_nemo_adapter.collections.data.base import BaseDataModule
from hyperpod_nemo_adapter.collections.data.datasets import (
    HuggingFacePretrainingDataset,
    HuggingFacePretrainingVisionDataset,
)
from hyperpod_nemo_adapter.utils.log_utils import Logger

_logger = Logger().get_logger()

print("=" * 80)
print(f"Flash Attention available: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Transformers version: {transformers.__version__}")
# Check if FA2 is installed
try:
    import flash_attn
    print(f"Flash Attention 2 installed: {flash_attn.__version__}")
except ImportError:
    print("Flash Attention 2 not installed")
print("=" * 80)

def mm_collate_fn(examples):
    lis = list(examples[0].keys())
    batch = {}
    for k in lis:
        if k == "pixel_values":
            batch[k] = torch.concat([torch.as_tensor(sample[k]) for sample in examples], dim=0)
        else:
            batch[k] = torch.stack([torch.as_tensor(sample[k]) for sample in examples], dim=0)
    return batch


class HuggingFaceDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig, trainer: Trainer, collate_fn=None):
        # Try config first, then fall back to env var
        use_packing_from_config = cfg.model.data.get("use_sequence_packing", None)
        use_packing_from_env = os.environ.get("USE_SEQUENCE_PACKING", "false").lower() == "true"
        
        self.use_packing = use_packing_from_config if use_packing_from_config is not None else use_packing_from_env
        self.tokenizer = None
        
        print("=" * 80)
        print("SEQUENCE PACKING LOADED - Omar's Custom Code")
        print(f"Training dir: {cfg.model.data.train_dir}")
        print(f"use_packing (from config): {use_packing_from_config}")
        print(f"use_packing (from env): {use_packing_from_env}")
        print(f"use_packing (final): {self.use_packing}")
        print("=" * 80)
        
        if collate_fn is None:
            if self.use_packing:
                _logger.info("="*80)
                _logger.info("--> SEQUENCE PACKING ENABLED")
                _logger.info("="*80)
                
                from transformers import DataCollatorWithFlattening, AutoTokenizer
                
                tokenizer_path = cfg.model.get("hf_model_name_or_path", "meta-llama/Llama-3.1-70B")
                access_token = cfg.model.get("hf_access_token", None)
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    token=access_token
                )
                
                collate_fn = DataCollatorWithFlattening(
                    tokenizer=self.tokenizer,
                    return_position_ids=True,
                    return_flash_attn_kwargs=True,
                )
                
                _logger.info(f"   DataCollatorWithFlattening initialized")
                _logger.info(f"   Tokenizer: {tokenizer_path}")
                _logger.info(f"   EOS token: {self.tokenizer.eos_token_id}")
            if self.use_packing:                
                from transformers import AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model.get("hf_model_name_or_path"),
                    token=cfg.model.get("hf_access_token")
                )
                
                # PyTorch CrossEntropyLoss convention
                IGNORE_INDEX = -100  
                EOS_TOKEN_ID = self.tokenizer.eos_token_id
                
                def collate_packed_sequences(examples):
                    batch = {
                        "input_ids": torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long),
                        "attention_mask": torch.tensor([ex["attention_mask"] for ex in examples], dtype=torch.long),
                    }
                    labels = batch["input_ids"].clone()
                    # Mask padding tokens (where attention_mask == 0)
                    labels[batch["attention_mask"] == 0] = IGNORE_INDEX
                    # Prevents the model from learning to predict tokens after a document ends enforcing  
                    # document boundaries so the model doesn't hallucinate relationships between unrelated 
                    # documents: P(tok | document_context)
                    labels[labels == EOS_TOKEN_ID] = IGNORE_INDEX
                    batch["labels"] = labels
                    return batch
                
                collate_fn = collate_packed_sequences
                
                _logger.info("="*80)
                _logger.info("  Sequence packing enabled")
                _logger.info(f" Tokenizer: {cfg.model.get('hf_model_name_or_path')}")
                _logger.info(f" EOS token: {self.tokenizer.eos_token_id}")
                _logger.info("="*80)                
            else:
                collate_fn = default_data_collator
                _logger.info("Using default data collator (no sequence packing)")
        
        super().__init__(cfg=cfg, trainer=trainer, collate_fn=collate_fn)

    def train_dataloader(self):
        input_path = self.cfg.model.data.train_dir
        trainset = HuggingFacePretrainingDataset(input_path=input_path, partition="train")
        self._train_ds = trainset.dataset
        return self._build_dataloader(self._train_ds, batch_size=self.cfg.model.train_batch_size)

    def val_dataloader(self):
        val_dir = self.cfg.model.data.val_dir
        if not val_dir:
            return None
        valset = HuggingFacePretrainingDataset(input_path=val_dir, partition="val")
        self._validation_ds = valset.dataset
        return self._build_dataloader(self._validation_ds, batch_size=self.cfg.model.val_batch_size)

    def get_batch(self, data):
        return data["input_ids"], data["attention_mask"], data["labels"]

    def get_val_batch(self, data):
        return self.get_batch(data)


class HuggingFaceMultiModalDataModule(HuggingFaceDataModule):
    """
    Lightning DataModule for HuggingFace Pretraining dataset pipelining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer, collate_fn=mm_collate_fn)

    def train_dataloader(self):
        input_path = self.cfg.model.data.train_dir
        trainset = HuggingFacePretrainingVisionDataset(input_path=input_path, partition="train")
        self._train_ds = trainset.dataset
        return self._build_dataloader(self._train_ds, batch_size=self.cfg.model.train_batch_size)

    def val_dataloader(self):
        val_dir = self.cfg.model.data.val_dir
        if not val_dir:
            return None
        valset = HuggingFacePretrainingVisionDataset(input_path=val_dir, partition="val")
        self._validation_ds = valset.dataset
        return self._build_dataloader(self._validation_ds, batch_size=self.cfg.model.val_batch_size)

    def get_batch(self, data):
        return data["input_ids"], data["attention_mask"], data["pixel_values"], data["labels"]


