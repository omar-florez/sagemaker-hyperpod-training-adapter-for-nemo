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

This module provides data loading and collation for pre-training with support for:
1. Standard mode: Returns (input_ids, attention_mask, labels)
2. Sequence packing mode: Returns (input_ids, attention_mask_4d, labels, position_ids, cu_seqlens_list, max_seqlen)

When use_smp_model=False:
  - Uses position_ids (reset per document) and 4D attention mask for HuggingFace models
  
When use_smp_model=True:
  - Uses cu_seqlens for TransformerEngine DotProductAttention hooks
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, Dataset

# Try to import datasets library
try:
    import datasets
    from datasets import load_from_disk
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Try to import transformers
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Constant for ignored labels in loss computation
IGNORE_INDEX = -100


class HFDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for HuggingFace datasets with sequence packing support.
    
    Supports two modes:
    1. Standard mode: Simple batching with padding
    2. Sequence packing mode: Multiple documents packed into sequences with proper masking
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        trainer: Trainer,
    ):
        super().__init__()
        self.cfg = cfg
        self.trainer = trainer
        self.tokenizer = None
        
        # Data configuration
        self.data_cfg = cfg.model.data if hasattr(cfg.model, 'data') else cfg.data
        self.train_dir = self.data_cfg.get("train_dir", None)
        self.val_dir = self.data_cfg.get("val_dir", None)
        
        # Batch sizes
        self.train_batch_size = cfg.model.get("train_batch_size", 1)
        self.val_batch_size = cfg.model.get("val_batch_size", 1)
        
        # Sequence length
        self.max_context_width = cfg.model.get("max_context_width", 8192)
        
        # Sequence packing configuration
        use_packing_from_config = self.data_cfg.get("use_sequence_packing", None)
        use_packing_from_env = os.environ.get("USE_SEQUENCE_PACKING", "false").lower() == "true"
        self._use_packing = use_packing_from_config if use_packing_from_config is not None else use_packing_from_env
        
        # SMP mode affects how we handle packing
        self._use_smp_model = cfg.get("use_smp_model", True)
        
        # EOS token ID (will be set from tokenizer or default to Llama 3.1)
        self._eos_token_id = 128001  # Default for Llama 3.1
        
        # Datasets
        self._train_ds = None
        self._val_ds = None
        self._train_dl = None
        self._val_dl = None
        
        # Logging counter
        self._log_count = 0
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if dist.get_rank() == 0:
            print(f"[HFDataModule] Setting up data module")
            print(f"  use_sequence_packing: {self._use_packing}")
            print(f"  use_smp_model: {self._use_smp_model}")
            print(f"  train_dir: {self.train_dir}")
            print(f"  max_context_width: {self.max_context_width}")
        
        # Load tokenizer if path provided
        hf_model_path = self.cfg.model.get("hf_model_name_or_path", None)
        if hf_model_path and HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
                if self.tokenizer.eos_token_id is not None:
                    self._eos_token_id = self.tokenizer.eos_token_id
                if dist.get_rank() == 0:
                    print(f"  Loaded tokenizer, EOS token ID: {self._eos_token_id}")
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"  Could not load tokenizer: {e}, using default EOS ID: {self._eos_token_id}")
        
        # Load datasets
        if self.train_dir and HAS_DATASETS:
            try:
                self._train_ds = load_from_disk(self.train_dir)
                if dist.get_rank() == 0:
                    print(f"  Loaded training dataset: {len(self._train_ds)} samples")
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"  Error loading training dataset: {e}")
        
        if self.val_dir and HAS_DATASETS:
            try:
                self._val_ds = load_from_disk(self.val_dir)
                if dist.get_rank() == 0:
                    print(f"  Loaded validation dataset: {len(self._val_ds)} samples")
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"  Error loading validation dataset: {e}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self._train_ds is None:
            raise ValueError("Training dataset not loaded. Call setup() first.")
        
        # Select appropriate collator
        if self._use_packing:
            collate_fn = self._collate_packed_sequences
        else:
            collate_fn = self._collate_standard
        
        self._train_dl = DataLoader(
            self._train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.data_cfg.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        return self._train_dl
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self._val_ds is None:
            return None
        
        # Select appropriate collator
        if self._use_packing:
            collate_fn = self._collate_packed_sequences
        else:
            collate_fn = self._collate_standard
        
        self._val_dl = DataLoader(
            self._val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.data_cfg.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
        return self._val_dl
    
    def get_batch(self, batch: Any) -> Tuple:
        """
        Extract tensors from batch.
        
        This method is called by the model's training_step to unpack the batch.
        
        Returns:
            Standard mode: (input_ids, attention_mask, labels)
            Packing mode: (input_ids, attention_mask, labels, position_ids, cu_seqlens_list, max_seqlen)
        """
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            labels = batch.get("labels", input_ids.clone())
            position_ids = batch.get("position_ids", None)
            cu_seqlens_list = batch.get("cu_seqlens_list", None)
            max_seqlen = batch.get("max_seqlen", None)
            
            if self._use_packing:
                return (input_ids, attention_mask, labels, position_ids, cu_seqlens_list, max_seqlen)
            else:
                return (input_ids, attention_mask, labels)
        
        elif isinstance(batch, (list, tuple)):
            if self._use_packing and len(batch) >= 6:
                return tuple(batch[:6])
            elif len(batch) >= 3:
                return tuple(batch[:3])
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)} with length {len(batch)}")
        
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")
    
    # =========================================================================
    # STANDARD COLLATOR (no packing)
    # =========================================================================
    
    def _collate_standard(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Standard collation without sequence packing.
        
        Expects samples to have 'input_ids' field.
        """
        input_ids_list = []
        
        for sample in samples:
            if isinstance(sample, dict):
                ids = sample.get("input_ids", sample.get("tokens", None))
            else:
                ids = sample
            
            if ids is None:
                raise ValueError(f"Sample missing 'input_ids' field: {sample.keys() if isinstance(sample, dict) else type(sample)}")
            
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long)
            elif isinstance(ids, torch.Tensor):
                ids = ids.long()
            
            input_ids_list.append(ids)
        
        # Pad to max length in batch
        max_len = min(max(len(ids) for ids in input_ids_list), self.max_context_width)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for ids in input_ids_list:
            # Truncate if needed
            ids = ids[:max_len]
            
            # Pad if needed
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
                mask = torch.cat([torch.ones(max_len - pad_len), torch.zeros(pad_len)])
            else:
                mask = torch.ones(max_len)
            
            # Labels: shift will happen in model, mask padding
            labels = ids.clone()
            labels[mask == 0] = IGNORE_INDEX
            
            batch_input_ids.append(ids)
            batch_attention_mask.append(mask)
            batch_labels.append(labels)
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }
    
    # =========================================================================
    # SEQUENCE PACKING COLLATOR
    # =========================================================================
    
    def _collate_packed_sequences(self, samples: List[Dict]) -> Dict[str, Any]:
        """
        Collate packed sequences with proper position_ids, attention_mask, and cu_seqlens.
        
        For each sample (which contains multiple packed documents):
        1. Find content_length from attention_mask (where padding starts)
        2. Find document boundaries using EOS tokens within content region
        3. Create position_ids that reset per document
        4. Create 4D block-diagonal attention mask (for HuggingFace models)
        5. Create cu_seqlens (for TransformerEngine with SMP)
        6. Mask labels appropriately (EOS tokens and padding get IGNORE_INDEX)
        
        NOTE: Padding tokens may also be EOS (128001), so we MUST use attention_mask
        to determine content_length before finding document boundaries.
        
        Returns dict with:
            - input_ids: [batch_size, seq_len]
            - attention_mask: [batch_size, 1, seq_len, seq_len] (4D block-diagonal)
            - labels: [batch_size, seq_len] with IGNORE_INDEX for EOS/padding
            - position_ids: [batch_size, seq_len] resetting per document
            - cu_seqlens_list: List of cu_seqlens tensors, one per sample
            - max_seqlen: Maximum document length in batch
        """
        batch_input_ids = []
        batch_attention_mask_4d = []
        batch_labels = []
        batch_position_ids = []
        batch_cu_seqlens = []
        batch_max_seqlen = 0
        
        for sample_idx, sample in enumerate(samples):
            # Extract input_ids and attention_mask
            if isinstance(sample, dict):
                ids = sample.get("input_ids", sample.get("tokens", None))
                orig_attention_mask = sample.get("attention_mask", None)
            else:
                ids = sample
                orig_attention_mask = None
            
            if ids is None:
                raise ValueError(f"Sample missing 'input_ids' field")
            
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long)
            elif isinstance(ids, torch.Tensor):
                ids = ids.long()
            
            if orig_attention_mask is not None:
                if isinstance(orig_attention_mask, list):
                    orig_attention_mask = torch.tensor(orig_attention_mask, dtype=torch.long)
                elif isinstance(orig_attention_mask, torch.Tensor):
                    orig_attention_mask = orig_attention_mask.long()
            
            seq_len = len(ids)
            
            # Truncate or pad to max_context_width
            if seq_len > self.max_context_width:
                ids = ids[:self.max_context_width]
                if orig_attention_mask is not None:
                    orig_attention_mask = orig_attention_mask[:self.max_context_width]
                seq_len = self.max_context_width
            elif seq_len < self.max_context_width:
                # Pad with EOS tokens (matching the data format)
                pad_len = self.max_context_width - seq_len
                ids = torch.cat([ids, torch.full((pad_len,), self._eos_token_id, dtype=torch.long)])
                if orig_attention_mask is not None:
                    orig_attention_mask = torch.cat([orig_attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                else:
                    # Create attention mask: 1 for original content, 0 for padding
                    orig_attention_mask = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            
            # CRITICAL: Find content_length from attention_mask FIRST
            # This is necessary because padding tokens are also EOS (128001)
            if orig_attention_mask is not None:
                # Content length = number of 1s in attention_mask (before padding starts)
                padding_positions = (orig_attention_mask == 0).nonzero(as_tuple=True)[0]
                if len(padding_positions) > 0:
                    content_length = padding_positions[0].item()
                else:
                    content_length = self.max_context_width
            else:
                # No attention_mask provided - assume all content
                content_length = self.max_context_width
            
            # Find EOS positions ONLY within content region (not padding)
            all_eos_positions = (ids == self._eos_token_id).nonzero(as_tuple=True)[0].tolist()
            eos_positions = [pos for pos in all_eos_positions if pos < content_length]
            
            # Build document boundaries for cu_seqlens
            # cu_seqlens = [0, end_of_doc1, end_of_doc2, ...]
            doc_boundaries = [0]
            for eos_pos in eos_positions:
                doc_boundaries.append(eos_pos + 1)
            
            # Ensure we have a final boundary at content_length
            # (in case last document doesn't end with EOS)
            if doc_boundaries[-1] != content_length:
                if content_length > doc_boundaries[-1]:
                    doc_boundaries.append(content_length)
            
            cu_seqlens = torch.tensor(doc_boundaries, dtype=torch.int32)
            
            # Calculate max document length in this sample
            for i in range(len(doc_boundaries) - 1):
                doc_len = doc_boundaries[i + 1] - doc_boundaries[i]
                batch_max_seqlen = max(batch_max_seqlen, doc_len)
            
            # Create position_ids that reset per document
            position_ids = torch.zeros(self.max_context_width, dtype=torch.long)
            for i in range(len(doc_boundaries) - 1):
                start = doc_boundaries[i]
                end = doc_boundaries[i + 1]
                doc_len = end - start
                position_ids[start:end] = torch.arange(doc_len)
            # Padding positions get 0 (doesn't matter as they're masked)
            
            # Create 4D block-diagonal attention mask
            # Shape: [1, seq_len, seq_len] -> will be expanded to [1, 1, seq_len, seq_len]
            attention_mask_4d = self._create_block_diagonal_mask(
                doc_boundaries, self.max_context_width
            )
            
            # Create labels with IGNORE_INDEX for:
            # 1. All EOS tokens
            # 2. All padding tokens
            labels = ids.clone()
            
            # Mask EOS tokens
            for eos_pos in eos_positions:
                labels[eos_pos] = IGNORE_INDEX
            
            # Mask padding (positions >= content_length)
            labels[content_length:] = IGNORE_INDEX
            
            # Store in batch
            batch_input_ids.append(ids)
            batch_attention_mask_4d.append(attention_mask_4d)
            batch_labels.append(labels)
            batch_position_ids.append(position_ids)
            batch_cu_seqlens.append(cu_seqlens)
        
        # Stack tensors
        input_ids = torch.stack(batch_input_ids)  # [batch, seq_len]
        attention_mask = torch.stack(batch_attention_mask_4d)  # [batch, 1, seq_len, seq_len]
        labels = torch.stack(batch_labels)  # [batch, seq_len]
        position_ids = torch.stack(batch_position_ids)  # [batch, seq_len]
        
        # Debug logging for first few batches
        if self._log_count < 3 and dist.is_initialized() and dist.get_rank() == 0:
            self._log_batch_info(
                input_ids, attention_mask, labels, position_ids,
                batch_cu_seqlens, batch_max_seqlen
            )
            self._log_count += 1
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
            "cu_seqlens_list": batch_cu_seqlens,
            "max_seqlen": batch_max_seqlen,
        }
    
    def _create_block_diagonal_mask(
        self,
        doc_boundaries: List[int],
        seq_len: int
    ) -> torch.Tensor:
        """
        Create a 4D block-diagonal causal attention mask.
        
        Each document can only attend to tokens within itself (causal).
        Documents cannot attend across boundaries.
        
        Args:
            doc_boundaries: List of document boundary positions [0, end1, end2, ...]
            seq_len: Total sequence length
            
        Returns:
            Tensor of shape [1, seq_len, seq_len] with:
            - 0 where attention is allowed
            - -inf (large negative) where attention is blocked
        """
        # Start with all blocked (will use additive mask)
        # For HF models: 0 = attend, -inf = don't attend
        mask = torch.full((1, seq_len, seq_len), float('-inf'))
        
        # Fill in causal blocks for each document
        for i in range(len(doc_boundaries) - 1):
            start = doc_boundaries[i]
            end = doc_boundaries[i + 1]
            
            # Create causal mask for this document block
            # Position j can attend to positions [start, j] (inclusive)
            for j in range(start, end):
                # Allow attention from position j to positions [start, j]
                mask[0, j, start:j+1] = 0.0
        
        return mask
    
    def _log_batch_info(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens_list: List[torch.Tensor],
        max_seqlen: int
    ):
        """Log batch information for debugging."""
        print("=" * 80)
        print(f"[HFDataModule] BATCH INFO (batch {self._log_count})")
        print("=" * 80)
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  attention_mask shape: {attention_mask.shape}")
        print(f"  labels shape: {labels.shape}")
        print(f"  position_ids shape: {position_ids.shape}")
        print(f"  max_seqlen: {max_seqlen}")
        print(f"  num samples with cu_seqlens: {len(cu_seqlens_list)}")
        
        # Log first sample details
        if len(cu_seqlens_list) > 0:
            sample_0_cu = cu_seqlens_list[0]
            print(f"\n  Sample 0:")
            print(f"    cu_seqlens: {sample_0_cu.tolist()}")
            print(f"    num_documents: {len(sample_0_cu) - 1}")
            
            # Show position_ids around first few document boundaries
            pos_ids = position_ids[0].tolist()
            boundaries = sample_0_cu.tolist()
            print(f"    position_ids around boundaries:")
            for i, bound in enumerate(boundaries[:4]):
                start = max(0, bound - 2)
                end = min(len(pos_ids), bound + 3)
                print(f"      boundary {i} (pos {bound}): ...{pos_ids[start:end]}...")
            
            # Count masked labels
            num_masked = (labels[0] == IGNORE_INDEX).sum().item()
            print(f"    labels masked (IGNORE_INDEX): {num_masked}")
            
            # Show attention mask structure
            print(f"    attention_mask: min={attention_mask[0].min().item():.2f}, max={attention_mask[0].max().item():.2f}")
            
            # Count non-masked positions in attention for first doc
            if len(boundaries) > 1:
                doc_end = boundaries[1]
                doc_mask = attention_mask[0, 0, :doc_end, :doc_end]
                non_masked = (doc_mask == 0).sum().item()
                expected = doc_end * (doc_end + 1) // 2  # Causal mask
                print(f"    First doc attention: {non_masked} non-masked (expected ~{expected} for causal)")
        
        print("=" * 80)


# =============================================================================
# OPTIONAL: Arrow Dataset Wrapper for pre-tokenized data
# =============================================================================

class ArrowDataset(Dataset):
    """
    Dataset wrapper for Arrow format pre-tokenized data.
    
    Expects data to have 'input_ids' column.
    """
    
    def __init__(self, data_path: str):
        if not HAS_DATASETS:
            raise ImportError("datasets library required for ArrowDataset")
        
        self.dataset = load_from_disk(data_path)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        
        input_ids = sample.get("input_ids", sample.get("tokens", None))
        if input_ids is None:
            raise ValueError(f"Sample {idx} missing 'input_ids' field")
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        return {"input_ids": input_ids}