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
import transformers

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
        use_packing_from_config = cfg.model.data.get("use_sequence_packing", None)
        use_packing_from_env = os.environ.get("USE_SEQUENCE_PACKING", "false").lower() == "true"
        
        self.use_packing = use_packing_from_config if use_packing_from_config is not None else use_packing_from_env
        self.tokenizer = None
        
        print("=" * 80)
        print("SEQUENCE PACKING CONFIGURATION")
        print(f"use_packing (config): {use_packing_from_config}")
        print(f"use_packing (env): {use_packing_from_env}")
        print(f"use_packing (final): {self.use_packing}")
        print("=" * 80)
        
        if collate_fn is None:
            if self.use_packing:
                from transformers import AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model.get("hf_model_name_or_path"),
                    token=cfg.model.get("hf_access_token")
                )
                
                IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss convention
                EOS_TOKEN_ID = self.tokenizer.eos_token_id
                
                # ===== FEATURE FLAGS =====
                USE_POSITION_RESET = True       # Toggle position ID resets
                USE_MASKED_LOSS = True          # Toggle loss masking for EOS/padding
                USE_BLOCK_DIAGONAL = True      # Toggle document-level attention blocking
                
                def collate_packed_sequences(examples):
                    """
                    Collator for pre-packed sequences with configurable features:
                    - Position ID resets at document boundaries (USE_POSITION_RESET)
                    - Loss masking for EOS and padding (USE_MASKED_LOSS)
                    - Document-level attention blocking (USE_BLOCK_DIAGONAL)
                    """
                    # Store original 2D attention mask (needed for label masking)
                    original_attention_mask = torch.tensor([ex["attention_mask"] for ex in examples], dtype=torch.long)
                    
                    batch = {
                        "input_ids": torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long),
                        "attention_mask": original_attention_mask.clone(),  # Will be overwritten if block diagonal
                    }
                    
                    batch_size, seq_len = batch["input_ids"].shape
                    
                    # ===== FEATURE 1: POSITION ID RESETS =====
                    if USE_POSITION_RESET:
                        if not hasattr(collate_packed_sequences, '_logged_pos_reset'):
                            print("  [ACTIVE] Generating position IDs with resets at EOS boundaries")
                            collate_packed_sequences._logged_pos_reset = True
                        
                        position_ids = torch.zeros_like(batch["input_ids"])
                        
                        for i in range(batch_size):
                            input_ids = batch["input_ids"][i]
                            attention_mask = original_attention_mask[i]  # Use original 2D mask
                            
                            # Find real content length
                            real_content_length = attention_mask.sum().item()
                            real_input_ids = input_ids[:real_content_length]
                            
                            # Find EOS positions in real content
                            eos_mask = (real_input_ids == EOS_TOKEN_ID)
                            eos_positions = eos_mask.nonzero(as_tuple=True)[0].tolist()
                            
                            # Generate position IDs with resets at EOS boundaries
                            last_eos = -1
                            for eos_pos in eos_positions:
                                length = eos_pos - last_eos
                                position_ids[i, last_eos+1:eos_pos+1] = torch.arange(length)
                                last_eos = eos_pos
                            
                            # Fill remaining real content after last EOS
                            if last_eos < real_content_length - 1:
                                length = real_content_length - last_eos - 1
                                position_ids[i, last_eos+1:real_content_length] = torch.arange(length)
                        
                        batch["position_ids"] = position_ids
                    else:
                        if not hasattr(collate_packed_sequences, '_logged_no_pos_reset'):
                            print("  [INACTIVE] Position resets disabled - using default continuous positions")
                            collate_packed_sequences._logged_no_pos_reset = True
                        # Don't add position_ids to batch
                        # Model will generate default continuous positions [0, 1, 2, ...]
                    
                    # ===== FEATURE 2: BLOCK DIAGONAL ATTENTION =====
                    if USE_BLOCK_DIAGONAL:
                        if not hasattr(collate_packed_sequences, '_logged_block_diag'):
                            print("  [ACTIVE] Generating document-level block diagonal attention mask")
                            collate_packed_sequences._logged_block_diag = True
                        
                        doc_attention_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool)
                        
                        for i in range(batch_size):
                            input_ids = batch["input_ids"][i]
                            attention_mask = original_attention_mask[i]  # Use original 2D mask
                            
                            # Find real content length
                            real_content_length = attention_mask.sum().item()
                            real_input_ids = input_ids[:real_content_length]
                            
                            # Find EOS positions in real content
                            eos_mask = (real_input_ids == EOS_TOKEN_ID)
                            eos_positions = eos_mask.nonzero(as_tuple=True)[0].tolist()
                            
                            # Create document-level attention blocks
                            doc_boundaries = [-1] + eos_positions + [real_content_length - 1]
                            
                            for doc_start, doc_end in zip(doc_boundaries[:-1], doc_boundaries[1:]):
                                start_pos = doc_start + 1
                                end_pos = doc_end + 1
                                
                                # Create causal mask within this document block
                                doc_len = end_pos - start_pos
                                causal_block = torch.tril(torch.ones((doc_len, doc_len), dtype=torch.bool))
                                doc_attention_mask[i, start_pos:end_pos, start_pos:end_pos] = causal_block
                        
                        batch["attention_mask"] = doc_attention_mask  # Overwrite with 3D block diagonal mask
                    else:
                        if not hasattr(collate_packed_sequences, '_logged_no_block_diag'):
                            print("  [INACTIVE] Block diagonal disabled - using simple 2D padding mask")
                            collate_packed_sequences._logged_no_block_diag = True
                        # Keep the original 2D attention mask (already in batch from line 79)
                    
                    # ===== FEATURE 3: MASKED LOSS =====
                    if USE_MASKED_LOSS:
                        if not hasattr(collate_packed_sequences, '_logged_masked_loss'):
                            print("  [ACTIVE] Masking loss for EOS tokens and padding")
                            collate_packed_sequences._logged_masked_loss = True
                        
                        labels = batch["input_ids"].clone()
                        
                        # ===== FIX: Always use original 2D mask for label masking =====
                        # Mask REAL EOS tokens first (document boundaries where attention_mask == 1)
                        real_eos_mask = (batch["input_ids"] == EOS_TOKEN_ID) & (original_attention_mask == 1)
                        labels[real_eos_mask] = IGNORE_INDEX
                        
                        # Mask padding tokens (where original attention_mask == 0)
                        labels[original_attention_mask == 0] = IGNORE_INDEX
                        
                        batch["labels"] = labels
                    else:
                        if not hasattr(collate_packed_sequences, '_logged_no_masked_loss'):
                            print("  [INACTIVE] Loss masking disabled - using raw input_ids as labels")
                            collate_packed_sequences._logged_no_masked_loss = True
                        # Use input_ids directly as labels (standard causal LM)
                        batch["labels"] = batch["input_ids"].clone()
                    
                    return batch
                
                collate_fn = collate_packed_sequences
                
                # ===== LOGGING CONFIGURATION =====
                _logger.info("="*80)
                _logger.info("SEQUENCE PACKING ENABLED")
                _logger.info(f" Tokenizer: {cfg.model.get('hf_model_name_or_path')}")
                _logger.info(f" EOS token ID: {self.tokenizer.eos_token_id}")
                _logger.info("")
                _logger.info("  Feature Flags:")
                if USE_POSITION_RESET:
                    _logger.info("    Position ID resets at document boundaries")
                else:
                    _logger.info("    Position ID resets DISABLED (using continuous positions)")
                
                if USE_BLOCK_DIAGONAL:
                    _logger.info("    Document-level block diagonal attention masking")
                else:
                    _logger.info("    Block diagonal attention DISABLED (using simple padding mask)")
                
                if USE_MASKED_LOSS:
                    _logger.info("    Loss masking for EOS and padding tokens")
                else:
                    _logger.info("    Loss masking DISABLED (using all tokens for loss)")
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
        """Extract batch data including position_ids"""
        if "position_ids" in data:
            return (
                data["input_ids"], 
                data["attention_mask"], 
                data["labels"],
                data["position_ids"]
            )
        return data["input_ids"], data["attention_mask"], data["labels"]

    def get_val_batch(self, data):
        return self.get_batch(data)


class HuggingFaceMultiModalDataModule(HuggingFaceDataModule):
    """Lightning DataModule for HuggingFace Pretraining dataset pipelining"""

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