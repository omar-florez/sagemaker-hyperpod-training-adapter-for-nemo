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

"""
HuggingFace Data Module with Sequence Packing Support

When USE_SEQUENCE_PACKING=true (env) or model.data.use_sequence_packing=true (config):
- Creates position_ids that reset per document
- Creates 4D block-diagonal attention mask  
- Returns 6-tuple: (input_ids, attention_mask, labels, position_ids, cu_seqlens_list, max_seqlen)

When disabled:
- Returns standard 3-tuple: (input_ids, attention_mask, labels)
"""

import os
from typing import List

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import default_data_collator

from hyperpod_nemo_adapter.collections.data.base import BaseDataModule
from hyperpod_nemo_adapter.collections.data.datasets import (
    HuggingFacePretrainingDataset,
    HuggingFacePretrainingVisionDataset,
)

# Constant for ignored labels in loss computation
IGNORE_INDEX = -100

# Default EOS token ID for Llama 3.1
DEFAULT_EOS_TOKEN_ID = 128001


def mm_collate_fn(examples):
    lis = list(examples[0].keys())
    batch = {}
    for k in lis:
        if k == "pixel_values":
            batch[k] = torch.concat([torch.as_tensor(sample[k]) for sample in examples], dim=0)
        else:
            batch[k] = torch.stack([torch.as_tensor(sample[k]) for sample in examples], dim=0)
    return batch


class SequencePackingCollator:
    """
    Collator for sequence packing that creates:
    - position_ids: Reset per document for proper RoPE
    - attention_mask: 4D block-diagonal mask for attention isolation
    - labels: Masked with IGNORE_INDEX for EOS tokens and padding
    - cu_seqlens: Cumulative sequence lengths for each document
    """
    
    def __init__(self, max_seq_length: int, eos_token_id: int = DEFAULT_EOS_TOKEN_ID):
        self.max_seq_length = max_seq_length
        self.eos_token_id = eos_token_id
        self._log_count = 0
    
    def __call__(self, samples: List[dict]) -> dict:
        """
        Collate packed sequences with proper position_ids, attention_mask, and cu_seqlens.
        
        NOTE: Padding tokens may also be EOS (128001), so we MUST use attention_mask
        to determine content_length before finding document boundaries.
        """
        batch_input_ids = []
        batch_attention_mask_4d = []
        batch_labels = []
        batch_position_ids = []
        batch_cu_seqlens = []
        batch_max_seqlen = 0
        
        for sample in samples:
            # Extract tensors from sample
            ids = sample.get("input_ids")
            orig_attention_mask = sample.get("attention_mask")
            labels = sample.get("labels")
            
            if ids is None:
                raise ValueError("Sample missing 'input_ids' field")
            
            # Convert to tensors if needed
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long)
            elif isinstance(ids, torch.Tensor):
                ids = ids.long().clone()
            
            if orig_attention_mask is not None:
                if isinstance(orig_attention_mask, list):
                    orig_attention_mask = torch.tensor(orig_attention_mask, dtype=torch.long)
                elif isinstance(orig_attention_mask, torch.Tensor):
                    orig_attention_mask = orig_attention_mask.long().clone()
            
            if labels is not None:
                if isinstance(labels, list):
                    labels = torch.tensor(labels, dtype=torch.long)
                elif isinstance(labels, torch.Tensor):
                    labels = labels.long().clone()
            else:
                labels = ids.clone()
            
            seq_len = len(ids)
            
            # Handle sequence length adjustments
            if seq_len > self.max_seq_length:
                ids = ids[:self.max_seq_length]
                labels = labels[:self.max_seq_length]
                if orig_attention_mask is not None:
                    orig_attention_mask = orig_attention_mask[:self.max_seq_length]
                seq_len = self.max_seq_length
            elif seq_len < self.max_seq_length:
                pad_len = self.max_seq_length - seq_len
                ids = torch.cat([ids, torch.full((pad_len,), self.eos_token_id, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
                if orig_attention_mask is not None:
                    orig_attention_mask = torch.cat([orig_attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                else:
                    orig_attention_mask = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            
            # CRITICAL: Find content_length from attention_mask FIRST
            # This is necessary because padding tokens are also EOS (128001)
            if orig_attention_mask is not None:
                padding_positions = (orig_attention_mask == 0).nonzero(as_tuple=True)[0]
                if len(padding_positions) > 0:
                    content_length = padding_positions[0].item()
                else:
                    content_length = self.max_seq_length
            else:
                content_length = self.max_seq_length
            
            # Find EOS positions ONLY within content region (not padding)
            all_eos_positions = (ids == self.eos_token_id).nonzero(as_tuple=True)[0].tolist()
            eos_positions = [pos for pos in all_eos_positions if pos < content_length]
            
            # Build document boundaries for cu_seqlens
            # cu_seqlens = [0, end_of_doc1, end_of_doc2, ...]
            doc_boundaries = [0]
            for eos_pos in eos_positions:
                doc_boundaries.append(eos_pos + 1)
            
            # Ensure final boundary at content_length
            if doc_boundaries[-1] != content_length:
                if content_length > doc_boundaries[-1]:
                    doc_boundaries.append(content_length)
            
            cu_seqlens = torch.tensor(doc_boundaries, dtype=torch.int32)
            
            # Calculate max document length in this sample
            for i in range(len(doc_boundaries) - 1):
                doc_len = doc_boundaries[i + 1] - doc_boundaries[i]
                batch_max_seqlen = max(batch_max_seqlen, doc_len)
            
            # Create position_ids that reset per document
            position_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
            for i in range(len(doc_boundaries) - 1):
                start = doc_boundaries[i]
                end = doc_boundaries[i + 1]
                doc_len = end - start
                position_ids[start:end] = torch.arange(doc_len)
            
            # Create 4D block-diagonal attention mask
            attention_mask_4d = self._create_block_diagonal_mask(doc_boundaries, self.max_seq_length)
            
            # Mask labels: EOS tokens and padding get IGNORE_INDEX
            for eos_pos in eos_positions:
                labels[eos_pos] = IGNORE_INDEX
            labels[content_length:] = IGNORE_INDEX
            
            # Store in batch
            batch_input_ids.append(ids)
            batch_attention_mask_4d.append(attention_mask_4d)
            batch_labels.append(labels)
            batch_position_ids.append(position_ids)
            batch_cu_seqlens.append(cu_seqlens)
        
        # Stack tensors
        result = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask_4d),
            "labels": torch.stack(batch_labels),
            "position_ids": torch.stack(batch_position_ids),
            "cu_seqlens_list": batch_cu_seqlens,
            "max_seqlen": batch_max_seqlen,
        }
        
        # Debug logging for first few batches
        if self._log_count < 3:
            try:
                if dist.is_initialized() and dist.get_rank() == 0:
                    self._log_batch_info(result)
            except Exception:
                pass
            self._log_count += 1
        
        return result
    
    def _create_block_diagonal_mask(self, doc_boundaries: List[int], seq_len: int) -> torch.Tensor:
        """
        Create 4D block-diagonal causal attention mask.
        
        Returns tensor of shape [1, seq_len, seq_len]:
        - 0 where attention is allowed
        - -inf where attention is blocked
        """
        mask = torch.full((1, seq_len, seq_len), float('-inf'))
        
        for i in range(len(doc_boundaries) - 1):
            start = doc_boundaries[i]
            end = doc_boundaries[i + 1]
            
            # Causal mask for this document block
            for j in range(start, end):
                mask[0, j, start:j+1] = 0.0
        
        return mask
    
    def _log_batch_info(self, batch: dict):
        """Log batch information for debugging."""
        print("=" * 80)
        print(f"[SequencePackingCollator] BATCH INFO (batch {self._log_count})")
        print("=" * 80)
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  position_ids shape: {batch['position_ids'].shape}")
        print(f"  max_seqlen: {batch['max_seqlen']}")
        print(f"  num samples with cu_seqlens: {len(batch['cu_seqlens_list'])}")
        
        if len(batch['cu_seqlens_list']) > 0:
            sample_0_cu = batch['cu_seqlens_list'][0]
            print(f"  Sample 0 cu_seqlens: {sample_0_cu.tolist()[:10]}...")
            print(f"  Sample 0 num_documents: {len(sample_0_cu) - 1}")
            
            # Show position_ids reset pattern
            pos_ids = batch['position_ids'][0].tolist()
            print(f"  Sample 0 position_ids first 30: {pos_ids[:30]}")
            
            # Count masked labels
            num_masked = (batch['labels'][0] == IGNORE_INDEX).sum().item()
            print(f"  Sample 0 labels masked: {num_masked}")
        
        print("=" * 80)


class HuggingFaceDataModule(BaseDataModule):
    """
    Lightning DataModule for HuggingFace Pretraining dataset pipelining
    
    Supports sequence packing when enabled via:
    - Environment variable: USE_SEQUENCE_PACKING=true
    - Config: model.data.use_sequence_packing=true
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer, collate_fn=None):
        # Check if sequence packing is enabled
        use_packing_from_config = cfg.model.data.get("use_sequence_packing", None) if hasattr(cfg.model, 'data') else None
        use_packing_from_env = os.environ.get("USE_SEQUENCE_PACKING", "false").lower() == "true"
        self._use_packing = use_packing_from_config if use_packing_from_config is not None else use_packing_from_env
        
        # Get max sequence length and EOS token ID
        self._max_seq_length = cfg.model.get("max_context_width", 8192)
        self._eos_token_id = cfg.model.get("eos_token_id", DEFAULT_EOS_TOKEN_ID)
        
        # Use appropriate collator
        if self._use_packing:
            collate_fn = SequencePackingCollator(
                max_seq_length=self._max_seq_length,
                eos_token_id=self._eos_token_id
            )
            print("=" * 80)
            print("SEQUENCE PACKING ENABLED in HuggingFaceDataModule")
            print(f"  max_seq_length: {self._max_seq_length}")
            print(f"  eos_token_id: {self._eos_token_id}")
            print("=" * 80)
        elif collate_fn is None:
            collate_fn = default_data_collator
        
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
        """
        Extract tensors from batch.
        
        Returns:
            Standard mode: (input_ids, attention_mask, labels)
            Packing mode: (input_ids, attention_mask, labels, position_ids, cu_seqlens_list, max_seqlen)
        """
        if self._use_packing:
            return (
                data["input_ids"],
                data["attention_mask"],
                data["labels"],
                data["position_ids"],
                data["cu_seqlens_list"],
                data["max_seqlen"],
            )
        else:
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