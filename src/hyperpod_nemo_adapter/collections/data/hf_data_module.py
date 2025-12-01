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
        
        print(f"self.use_packing: {self.use_packing}")
        
        if collate_fn is None:
            if self.use_packing:
                print("=" * 80)
                print(f"Flash Attention available: {torch.backends.cuda.flash_sdp_enabled()}")
                print(f"Transformers version: {transformers.__version__}")
                try:
                    import flash_attn
                    print(f"Flash Attention 2 installed: {flash_attn.__version__}")
                except ImportError:
                    print("Flash Attention 2 not installed")
                print("=" * 80)
                
                print("=" * 80)
                print("SEQUENCE PACKING CONFIGURATION (cu_seqlens mode)")
                print(f"use_packing (config): {use_packing_from_config}")
                print(f"use_packing (env): {use_packing_from_env}")
                print(f"use_packing (final): {self.use_packing}")
                print("=" * 80)
                
                from transformers import AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model.get("hf_model_name_or_path"),
                    token=cfg.model.get("hf_access_token")
                )
                
                IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss convention
                EOS_TOKEN_ID = self.tokenizer.eos_token_id
                
                def collate_packed_sequences_cu_seqlens(examples):
                    """
                    Collator for pre-packed sequences using cu_seqlens format.
                    
                    Computes:
                    - cu_seqlens_q/kv: Cumulative sequence lengths for each sample
                    - max_seqlen: Maximum document length in batch
                    - labels: With EOS and padding masked
                    
                    Does NOT compute position_ids or block diagonal masks - 
                    cu_seqlens handles attention blocking in TransformerEngine.
                    """
                    if not hasattr(collate_packed_sequences_cu_seqlens, '_logged'):
                        print("  [ACTIVE] cu_seqlens mode: Computing cumulative sequence lengths")
                        print(f"  [ACTIVE] EOS token ID: {EOS_TOKEN_ID}")
                        print("  [ACTIVE] Masking EOS and padding in labels")
                        collate_packed_sequences_cu_seqlens._logged = True
                    
                    # Stack into batch tensors
                    input_ids = torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long)
                    attention_mask = torch.tensor([ex["attention_mask"] for ex in examples], dtype=torch.long)
                    
                    batch_size, seq_len = input_ids.shape
                    
                    # ===== COMPUTE CU_SEQLENS FOR EACH SAMPLE =====
                    # List of cu_seqlens tensors, one per sample in batch
                    cu_seqlens_list = []
                    max_seqlen_in_batch = 0
                    
                    for b in range(batch_size):
                        # Find real content length (excluding padding)
                        real_length = attention_mask[b].sum().item()
                        sample_ids = input_ids[b, :real_length]
                        
                        # Find EOS positions (document boundaries)
                        eos_mask = (sample_ids == EOS_TOKEN_ID)
                        eos_positions = eos_mask.nonzero(as_tuple=True)[0]
                        
                        if len(eos_positions) > 0:
                            # cu_seqlens: [0, end_of_doc1, end_of_doc2, ...]
                            # Each document ends at EOS (inclusive), so positions are eos_pos + 1
                            cu_seqlens = torch.zeros(len(eos_positions) + 1, dtype=torch.int32)
                            cu_seqlens[1:] = eos_positions + 1
                            
                            # Check if there's content after the last EOS (incomplete doc)
                            last_eos = eos_positions[-1].item()
                            if last_eos + 1 < real_length:
                                # Add the remaining content as final segment
                                cu_seqlens = torch.cat([
                                    cu_seqlens, 
                                    torch.tensor([real_length], dtype=torch.int32)
                                ])
                        else:
                            # No EOS found - treat entire sequence as one document
                            cu_seqlens = torch.tensor([0, real_length], dtype=torch.int32)
                        
                        cu_seqlens_list.append(cu_seqlens)
                        
                        # Compute document lengths and track max
                        doc_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                        if len(doc_lengths) > 0:
                            max_seqlen_in_batch = max(max_seqlen_in_batch, doc_lengths.max().item())
                    
                    # ===== CREATE LABELS WITH MASKING =====
                    labels = input_ids.clone()
                    
                    # Mask EOS tokens (document boundaries) - don't predict EOS
                    eos_mask = (input_ids == EOS_TOKEN_ID) & (attention_mask == 1)
                    labels[eos_mask] = IGNORE_INDEX
                    
                    # Mask padding tokens
                    labels[attention_mask == 0] = IGNORE_INDEX
                    
                    # ===== BUILD BATCH DICT =====
                    batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,  # Keep 2D mask for compatibility
                        "labels": labels,
                        # cu_seqlens for each sample in batch
                        "cu_seqlens_list": cu_seqlens_list,
                        "max_seqlen": max_seqlen_in_batch,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                    }
                    
                    return batch
                
                collate_fn = collate_packed_sequences_cu_seqlens
                
                # ===== LOGGING CONFIGURATION =====
                _logger.info("=" * 80)
                _logger.info("SEQUENCE PACKING ENABLED (cu_seqlens mode)")
                _logger.info(f"  Tokenizer: {cfg.model.get('hf_model_name_or_path')}")
                _logger.info(f"  EOS token ID: {self.tokenizer.eos_token_id}")
                _logger.info("")
                _logger.info("  Features:")
                _logger.info("    - cu_seqlens computed from EOS positions")
                _logger.info("    - DotProductAttention hooks inject cu_seqlens")
                _logger.info("    - Labels masked for EOS and padding")
                _logger.info("    - Position IDs: default continuous (RoPE not patched)")
                _logger.info("=" * 80)
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
        """
        Extract batch data including cu_seqlens for sequence packing.
        
        Returns:
            For packing mode: (input_ids, attention_mask, labels, cu_seqlens_list, max_seqlen)
            For standard mode: (input_ids, attention_mask, labels)
        """
        if "cu_seqlens_list" in data:
            return (
                data["input_ids"],
                data["attention_mask"],
                data["labels"],
                data["cu_seqlens_list"],
                data["max_seqlen"],
            )
        # Standard mode (no packing)
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