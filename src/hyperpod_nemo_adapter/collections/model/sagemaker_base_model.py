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

import inspect
import math
import os
import time
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.sagemaker as tsm
import transformer_engine
import transformers
from accelerate import init_empty_weights
from nemo.core.classes.modelPT import ModelPT
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from packaging import version as pversion
from peft import LoraConfig, PeftModel, get_peft_model
from pytorch_lightning import Trainer
from torch.sagemaker import transform
from torch.sagemaker.context_parallel.utils import setup_transformer_engine_cp_groups
from torch.sagemaker.grad_norm import clip_grad_norm_
from torch.sagemaker.moe.moe_config import MoEConfig
from torch.sagemaker.utils.process_group_utils import get_global_ranks
from transformer_engine.common.recipe import DelayedScaling, Format
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from hyperpod_nemo_adapter.constants import CONFIG_MAPPING_HF_TO_RECIPE_ALIASES
from hyperpod_nemo_adapter.patches import patch_llama_flash_attn_cp
from hyperpod_nemo_adapter.utils.config_utils import get_hf_config_from_name_or_path
from hyperpod_nemo_adapter.utils.dpo_utils import compute_dpo_loss
from hyperpod_nemo_adapter.utils.general_utils import can_use_multimodal
from hyperpod_nemo_adapter.utils.log_utils import Logger
from hyperpod_nemo_adapter.utils.train_utils import (
    compute_tflops,
    get_batch_for_cp_rank,
)

if can_use_multimodal():
    from transformers import MllamaForConditionalGeneration

    from hyperpod_nemo_adapter.patches import patch_mllama_dtype

TF_VERSION = pversion.parse(transformers.__version__)

_logger = Logger().get_logger()


class SageMakerNLPBaseModel(ModelPT):
    """
    General Lightning Model class for SageMaker adapter, it deals with general model/optimizer setup
    and training/eval behaviors.
    User will need to either consume the provided inheritors or inherit and implement their own model class.
    """

    # Whether if the model is predefined
    # All subclass should set this to True
    predefined_model = False

    def __init__(self, cfg: DictConfig, trainer: Trainer, use_smp_model=True):
        self.grad_norm = None
        self._cfg = cfg
        self.model = None
        self.ref_model = None
        self.use_smp_model = use_smp_model
        self.model_config = None

        self.val_loss = 0

        self._config_mapping_hf_to_recipe_aliases = None

        self.set_config_mapping_hf_to_recipe_aliases()
        # Setup Transformer Engine Variable
        os.environ["NVTE_TORCH_COMPILE"] = "0"
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

        if self.do_patch_attn_context_parallel:
            # avoid error trying to access non-existent attribute in TE extra state
            # in `from_pretrained`
            os.environ["ACCELERATE_USE_FSDP"] = "True"
            os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "True"

        if self._cfg.get("nvte_attn_backend", None) is not None:
            if self._cfg.nvte_attn_backend == "fused":
                # use fused-attn backend
                os.environ["NVTE_FUSED_ATTN"] = "1"
                os.environ["NVTE_FLASH_ATTN"] = "0"
            elif self._cfg.nvte_attn_backend == "flash":
                # use flash-attn backend
                os.environ["NVTE_FUSED_ATTN"] = "0"
                os.environ["NVTE_FLASH_ATTN"] = "1"

        super().__init__(cfg, trainer)

    def get_model_config(self):
        """
        Get model config to build the model, should be implemented in specific model class
        """
        cls = type(self).__name__
        raise NotImplementedError(f"{cls}.get_model_config not implemented")

    def set_config_mapping_hf_to_recipe_aliases(self):
        # Map the hf args with recipe aliases, could be nemo/hf style
        self._config_mapping_hf_to_recipe_aliases = CONFIG_MAPPING_HF_TO_RECIPE_ALIASES

    def _get_model_configurable_dict(self):
        """
        Get the a dict that contains all configurable values of the current model
        This method will only be used for child class
        """
        config_dict = {}

        for hf_config, recipe_configs in self._config_mapping_hf_to_recipe_aliases.items():
            for recipe_alias in recipe_configs:
                if self._cfg.get(recipe_alias, None) is not None:
                    if type(self._cfg.get(recipe_alias)).__name__ == "DictConfig":
                        # Check nested config, like rope_scaling
                        config_dict[hf_config] = dict(self._cfg.get(recipe_alias))
                    else:
                        config_dict[hf_config] = self._cfg.get(recipe_alias)

        if dist.get_rank() == 0:
            _logger.info(f"Overriding model config with {config_dict}")
        return config_dict

    @property
    def use_peft(self):
        return self._cfg.get("peft", None) is not None and self._cfg.peft.get("peft_type", None) is not None

    @property
    def peft_type(self):
        return self._cfg.peft.get("peft_type", None)

    @property
    def do_finetune_with_pretrained_weights(self):
        """
        Returns true if we need to load pretrained weights from model_name_or_path
        """
        return (
            self._cfg.get("do_finetune", True)
            and self._cfg.get("hf_model_name_or_path", None) is not None
            and (self._cfg.get("resume_from_checkpoint", None) is None or self.use_peft)
        )

    @property
    def do_patch_attn_context_parallel(self):
        # support non-SMP context parallel usage.
        # this is mostly for llama 405b QLoRA CP.
        return not self.use_smp_model and self._cfg.get("context_parallel_degree", 1) > 1

    @property
    def do_patch_mllama(self):
        # https://github.com/huggingface/transformers/issues/34207
        return (
            not self.use_smp_model
            and self._cfg.get("model_type", None) == "llama_v3"
            and self._cfg.get("multi_modal", False)
        )

    def setup(self, *a, **kw):
        if self.do_patch_mllama:
            patch_mllama_dtype.apply_patch(dtype=torch.bfloat16 if self._cfg.precision == "bf16" else torch.float32)
        if self.do_patch_attn_context_parallel:
            patch_llama_flash_attn_cp.apply_patch()
        if not self.predefined_model:
            assert not self.use_smp_model, "model that is not predefined can not support use_smp_model=True"
            assert (
                self._cfg.get("hf_model_name_or_path", None) is not None
            ), "hf_model_name_or_path is required when the model is not predefined"
            _logger.info(
                f"{self._cfg.hf_model_name_or_path} is not a predefined model, most of smp features will be ignored, e.g. TP/fp8, only FSDP/activation_checkpoint can be applied."
            )
            # Using config from the pretrained model
            self.model_config = get_hf_config_from_name_or_path(self._cfg)
        else:
            self.model_config = self.get_model_config()
        # Disable KV cache for HF models
        if hasattr(self.model_config, "use_cache"):
            self.model_config.use_cache = False
        # Adding delayed_param config to HF model config
        self.dp_size = dist.get_world_size() // (
            self._cfg.get("context_parallel_degree", 1) * self._cfg.get("tensor_model_parallel_degree", 1)
        )
        self.model_config.delayed_param = self._cfg.delayed_param
        model = self._initialize_model(self.model_config)
        if self.do_patch_attn_context_parallel:
            # check that we are using patched attention for context parallel
            assert any(
                [submodule.__module__ == "transformer_engine.pytorch.attention" for submodule in model.modules()]
            ), "This model does not support context parallel with use_smp_model=False."
            # setup TransformerEngine CP groups
            setup_transformer_engine_cp_groups(
                model, get_global_ranks(tsm.state.cp_process_group), tsm.state.cp_process_group
            )
        if self.do_finetune_with_pretrained_weights:
            dist.barrier()
        if self.use_smp_model:
            self.model = self._transform(model)
        else:
            self.model = model
        if self._cfg.dpo.get("enabled", False) and not self.use_peft:
            ref_model = self._initialize_model(self.model_config)
            if self.do_patch_attn_context_parallel:
                setup_transformer_engine_cp_groups(
                    ref_model, get_global_ranks(tsm.state.cp_process_group), tsm.state.cp_process_group
                )
            if self.use_smp_model:
                self.ref_model = self._transform(ref_model)
            else:
                self.ref_model = ref_model
            self.ref_model.eval()

        self.fp8_recipe = self._fp8_delayed_scaling()

    def param_init_fn(self, module):
        _logger.warning(
            f"A _param_init_fn has not been implemented for the current model class. Proceeding to train with delayed_param={self._cfg.delayed_param} will lead to convergence issues."
        )
        return module.to_empty(device=torch.device("cuda"), recurse=False)

    def _fp8_delayed_scaling(self):
        if self.use_smp_model and self._cfg.fp8:
            return DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=self._cfg.fp8_amax_history_len,
                amax_compute_algo=self._cfg.fp8_amax_compute_algo,
            )

    def _transform(self, model):
        moe_config = None
        load_state_dict_from_rank0 = self.do_finetune_with_pretrained_weights
        if self._cfg.moe:
            moe_config = MoEConfig(
                smp_moe=self.use_smp_model,
                random_seed=12345,
                moe_load_balancing=self._cfg.moe_load_balancing,
                global_token_shuffle=self._cfg.global_token_shuffle,
                moe_all_to_all_dispatcher=self._cfg.moe_all_to_all_dispatcher,
                moe_aux_loss_coeff=0.001,
                moe_z_loss_coeff=0.001,
                use_cpu_initialization=self.do_finetune_with_pretrained_weights and dist.get_rank() == 0,
            )

        if self._cfg.moe and self._cfg.delayed_param and (not load_state_dict_from_rank0 or dist.get_rank() != 0):
            with init_empty_weights():
                return transform(
                    model,
                    config=moe_config,
                    load_state_dict_from_rank0=load_state_dict_from_rank0,
                )
        else:
            # Note: Current tsm transform() function only allows the config param to be used for MoEConfigs.
            return transform(
                model,
                config=moe_config,
                load_state_dict_from_rank0=load_state_dict_from_rank0,
            )

    def _initialize_model(self, model_cfg):
        if self._cfg.get("delayed_param", None) is None or not self._cfg.delayed_param:
            # initialize model on host memory
            return self.build_model(model_cfg)
        if self.do_finetune_with_pretrained_weights and dist.get_rank() == 0:
            # initialize model only on rank 0
            return self.build_model(model_cfg)
        with init_empty_weights():
            # initialize model on meta device
            return self.build_model(model_cfg)

    def get_quantization_config(self):
        if self._cfg.peft.get("peft_type", None) is not None and "qlora_4bit" in self._cfg.peft.peft_type:
            if self.do_patch_attn_context_parallel:
                # if patching attention with TransformerEngine CP, HF crashes on get_keys_to_not_convert.
                # instead, specify llm_int8_skip_modules, to bypass get_keys_to_not_convert function.
                llm_int8_skip_modules = ["lm_head"]
            else:
                llm_int8_skip_modules = None
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.bfloat16,
                llm_int8_skip_modules=llm_int8_skip_modules,
            )
        else:
            quantization_config = None
        return quantization_config

    def get_lora_config(self):
        target_modules = None
        if self._cfg.peft.get("target_modules", None) is not None:
            target_modules = list(self._cfg.peft.target_modules)
            if type(target_modules) is list and len(target_modules) == 1:
                target_modules = target_modules[0]
        lora_config = LoraConfig(
            target_modules=target_modules or "all-linear",
            # Alpha parameter for LoRA scaling
            lora_alpha=self._cfg.peft.alpha,
            # Dropout probability for LoRA layers
            lora_dropout=self._cfg.peft.dropout,
            # LoRA attention dimension
            r=self._cfg.peft.rank,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        return lora_config

    def build_model(self, model_cfg):
        if self.use_peft:
            return self._build_model_from_pretrain_peft(model_cfg)
        if self.do_finetune_with_pretrained_weights and dist.get_rank() == 0:
            return self._build_model_from_pretrain(model_cfg)
        return self._build_model(model_cfg)

    def _build_model_from_pretrain_peft(self, model_cfg):
        assert not self.use_smp_model, "Must set use_smp_model=False to use PEFT"
        assert self._cfg.do_finetune, "Must set do_finetune=True to use PEFT"
        assert self._cfg.hf_model_name_or_path is not None, "Must provide pretrained weights to use PEFT"

        # set env vars for efficient HF model loading (PEFT does not use SMP delayed param)
        # see https://github.com/huggingface/transformers/blob/27903de7ecfc21e9b5a061c46c3b1ff73539d385/src/transformers/modeling_utils.py#L140
        os.environ["ACCELERATE_USE_FSDP"] = "True"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "True"
        if (
            self.peft_type is not None
            and self.peft_type == "lora"
            and self._cfg.get("model_type", None) == "deepseek_r1"
        ):
            os.environ["FORCE_ACTIVATE"] = "True"

        quantization_config = self.get_quantization_config()

        model = self._build_model_from_pretrain(
            model_cfg, torch_dtype=torch.bfloat16, quantization_config=quantization_config
        )

        lora_config = self.get_lora_config()

        model.enable_input_require_grads()

        checkpoint_dir = self.trainer.strategy.cfg.exp_manager.resume_from_checkpoint
        if checkpoint_dir is not None:
            if dist.get_rank() == 0:
                _logger.debug(f"Model before loading adapter weights: {model}")
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
            if dist.get_rank() == 0:
                _logger.info(f"Loaded adapter weights from {checkpoint_dir}.")
                _logger.debug(f"Model after loading adapter weights: {model}")
        else:
            model = get_peft_model(model, lora_config)

        if dist.get_rank() == 0:
            model.print_trainable_parameters()

        return model

    def _build_model_from_pretrain(self, model_cfg, torch_dtype=None, quantization_config=None):
        path = self._cfg.hf_model_name_or_path
        _logger.info("Loading pretrained weights from %s.", path)
        use_flash_attn = self._cfg.use_flash_attention
        attn = "flash_attention_2"
        # TODO add support later for flash att
        # ValueError: MllamaForCausalLM does not support Flash Attention 2.0 yet
        if self._cfg.get("multi_modal", None):
            return MllamaForConditionalGeneration.from_pretrained(
                path, config=model_cfg, torch_dtype=torch_dtype, quantization_config=quantization_config
            )
        access_token = self._cfg.get("hf_access_token", None)
        if TF_VERSION < pversion.parse("4.37.1") or not use_flash_attn:
            return AutoModelForCausalLM.from_pretrained(
                path,
                config=model_cfg,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                token=access_token,
            )
        return AutoModelForCausalLM.from_pretrained(
            path,
            attn_implementation=attn,
            config=model_cfg,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            token=access_token,
        )

    def _build_model(self, model_cfg):
        use_flash_attn = self._cfg.use_flash_attention
        attn = "flash_attention_2"
        # TODO add support later for flash att
        # ValueError: MllamaForCausalLM does not support Flash Attention 2.0 yet
        if self._cfg.get("multi_modal", None):
            model = MllamaForConditionalGeneration(config=model_cfg)
            return model
        if TF_VERSION < pversion.parse("4.37.1") or not use_flash_attn:
            return AutoModelForCausalLM.from_config(model_cfg)
        return AutoModelForCausalLM.from_config(model_cfg, attn_implementation=attn)

    # =========================================================================
    # DEBUG METHODS FOR SEQUENCE PACKING
    # =========================================================================

    def _debug_log_batch_structure(self, batch_idx, input_ids, attention_mask, labels, position_ids, use_packing):
        """
        DEBUG STEP 1: Log complete batch structure to understand data flow.
        """
        print("=" * 100)
        print(f"DEBUG STEP 1: BATCH STRUCTURE (batch_idx={batch_idx})")
        print("=" * 100)

        batch_size, seq_len = input_ids.shape
        print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
        print(f"use_packing: {use_packing}")

        # Get tokenizer info
        try:
            tokenizer = self.trainer.datamodule.tokenizer
            print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
            eos_token_id = tokenizer.eos_token_id if tokenizer else 128001  # Llama-3 default
            print(f"EOS token ID: {eos_token_id}")
        except Exception:
            eos_token_id = 128001
            print(f"EOS token ID (default): {eos_token_id}")

        IGNORE_INDEX = -100

        for sample_idx in range(min(batch_size, 2)):  # Log first 2 samples
            print("-" * 80)
            print(f"SAMPLE {sample_idx}:")
            print("-" * 80)

            ids = input_ids[sample_idx].cpu()
            lab = labels[sample_idx].cpu()

            # Find EOS positions
            eos_positions = (ids == eos_token_id).nonzero(as_tuple=True)[0].tolist()
            print(f"  EOS positions: {eos_positions[:20]}{'...' if len(eos_positions) > 20 else ''}")
            print(f"  Number of documents: {len(eos_positions)}")

            # Find padding (attention_mask == 0)
            padding_start = seq_len
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask = attention_mask[sample_idx].cpu()
                    padding_positions = (mask == 0).nonzero(as_tuple=True)[0]
                    if len(padding_positions) > 0:
                        padding_start = padding_positions[0].item()
                elif attention_mask.dim() == 3:
                    # 3D mask - check diagonal for valid positions
                    mask_diag = attention_mask[sample_idx].diagonal().cpu()
                    padding_positions = (mask_diag == 0).nonzero(as_tuple=True)[0]
                    if len(padding_positions) > 0:
                        padding_start = padding_positions[0].item()

            print(f"  Padding starts at position: {padding_start}")
            print(f"  Real content length: {padding_start}")

            # Find IGNORE_INDEX positions in labels
            ignored_positions = (lab == IGNORE_INDEX).nonzero(as_tuple=True)[0].tolist()
            print(f"  Labels with IGNORE_INDEX (-100): {len(ignored_positions)} positions")

            # Show document boundaries with tokens (condensed format)
            print(f"")
            print(f"  Document structure:")
            doc_start = 0
            for doc_idx, eos_pos in enumerate(eos_positions[:5]):  # Show first 5 docs
                doc_end = eos_pos + 1
                doc_len = doc_end - doc_start

                # Get tokens for this document
                doc_input_ids = ids[doc_start:doc_end].tolist()
                doc_labels = lab[doc_start:doc_end].tolist()

                # Count ignored labels in this document
                ignored_in_doc = sum(1 for l in doc_labels if l == IGNORE_INDEX)

                print(f"    Doc {doc_idx}: positions [{doc_start}:{doc_end}] (len={doc_len}), ignored_labels={ignored_in_doc}")
                
                # Condensed format: first 5 ... last 5
                if len(doc_input_ids) > 10:
                    print(f"      input_ids:    [{', '.join(map(str, doc_input_ids[:5]))}, ..., {', '.join(map(str, doc_input_ids[-5:]))}]")
                    print(f"      labels:       [{', '.join(map(str, doc_labels[:5]))}, ..., {', '.join(map(str, doc_labels[-5:]))}]")
                else:
                    print(f"      input_ids:    {doc_input_ids}")
                    print(f"      labels:       {doc_labels}")

                if position_ids is not None:
                    pos = position_ids[sample_idx].cpu()
                    doc_positions = pos[doc_start:doc_end].tolist()
                    
                    if len(doc_positions) > 10:
                        print(f"      position_ids: [{', '.join(map(str, doc_positions[:5]))}, ..., {', '.join(map(str, doc_positions[-5:]))}]")
                    else:
                        print(f"      position_ids: {doc_positions}")

                    # Verify position IDs start at 0
                    if doc_positions[0] != 0:
                        print(f"      [WARNING] Position IDs do not start at 0!")
                    # Verify position IDs are sequential
                    expected = list(range(doc_len))
                    if doc_positions != expected:
                        print(f"      [WARNING] Position IDs are not sequential [0, 1, 2, ...]!")

                doc_start = doc_end

            # Show content after last EOS (if any)
            if eos_positions and eos_positions[-1] + 1 < padding_start:
                remaining_start = eos_positions[-1] + 1
                remaining_len = padding_start - remaining_start
                print(f"    Remaining content after last EOS: positions [{remaining_start}:{padding_start}] (len={remaining_len})")

            # Show alignment around EOS and MASKED positions (±1 context)
            print(f"")
            print(f"  Token alignment around EOS/MASKED positions (±1 context):")
            print(f"  {'Pos':>5} | {'InputID':>8} | {'Label':>8} | {'PosID':>6} | Notes")
            print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*30}")

            # Collect positions of interest: EOS positions and MASKED positions
            positions_of_interest = set()
            for eos_pos in eos_positions[:10]:  # First 10 EOS positions
                positions_of_interest.add(max(0, eos_pos - 1))
                positions_of_interest.add(eos_pos)
                positions_of_interest.add(min(seq_len - 1, eos_pos + 1))
            
            # Also add first few MASKED positions that aren't near EOS
            masked_positions = (lab == IGNORE_INDEX).nonzero(as_tuple=True)[0].tolist()
            for masked_pos in masked_positions[:20]:
                if masked_pos not in positions_of_interest:
                    positions_of_interest.add(max(0, masked_pos - 1))
                    positions_of_interest.add(masked_pos)
                    positions_of_interest.add(min(seq_len - 1, masked_pos + 1))

            # Sort and print
            sorted_positions = sorted(positions_of_interest)
            last_printed = -2
            for pos in sorted_positions:
                if pos >= seq_len:
                    continue
                    
                # Add separator if there's a gap
                if pos > last_printed + 1 and last_printed >= 0:
                    print(f"  {'...':>5} |")
                
                input_id = ids[pos].item()
                label = lab[pos].item()
                pos_id = position_ids[sample_idx, pos].item() if position_ids is not None else pos

                notes = []
                if input_id == eos_token_id:
                    notes.append("EOS")
                if label == IGNORE_INDEX:
                    notes.append("MASKED")
                if position_ids is not None and pos > 0:
                    prev_pos_id = position_ids[sample_idx, pos - 1].item()
                    if pos_id < prev_pos_id:
                        notes.append("POS_RESET")
                if pos in eos_positions:
                    notes.append("<-- DOC BOUNDARY")

                notes_str = ", ".join(notes) if notes else ""
                print(f"  {pos:>5} | {input_id:>8} | {label:>8} | {pos_id:>6} | {notes_str}")
                last_printed = pos

        # Log attention mask info
        if attention_mask is not None:
            print(f"")
            print(f"  Attention mask:")
            print(f"    Shape: {attention_mask.shape}")
            print(f"    Dtype: {attention_mask.dtype}")

            if attention_mask.dim() == 3:
                # Visualize block structure for first sample around first EOS
                mask = attention_mask[0].cpu()
                if eos_positions:
                    first_eos = eos_positions[0]
                    start_row = max(0, first_eos - 5)
                    end_row = min(seq_len, first_eos + 10)
                    print(f"    Block diagonal structure around first EOS (pos {first_eos}):")
                    header = "      " + "".join([f"{i % 10}" for i in range(start_row, end_row)])
                    print(header)
                    for row in range(start_row, end_row):
                        line = f"  {row:>3} "
                        for col in range(start_row, end_row):
                            if mask[row, col]:
                                line += "#"
                            else:
                                line += "."
                        if row in eos_positions[:3]:
                            line += " <-- EOS"
                        print(line)
                else:
                    print(f"    Block diagonal structure (20x20):")
                    header = "      " + "".join([f"{i % 10}" for i in range(20)])
                    print(header)
                    for row in range(min(20, seq_len)):
                        line = f"  {row:>3} "
                        for col in range(min(20, seq_len)):
                            if mask[row, col]:
                                line += "#"
                            else:
                                line += "."
                        print(line)

    def _debug_log_model_signature(self):
        """
        DEBUG STEP 2: Log model forward signature to verify what parameters are accepted.
        """
        print("=" * 100)
        print("DEBUG STEP 2: MODEL FORWARD SIGNATURE")
        print("=" * 100)

        # Check the wrapped model
        model = self.model
        print(f"Model type: {type(model)}")

        # Try to get the inner model
        if hasattr(model, '_fsdp_wrapped_module'):
            inner = model._fsdp_wrapped_module
            print(f"Inner model type: {type(inner)}")

            # Check forward signature
            try:
                sig = inspect.signature(inner.forward)
                params = list(sig.parameters.keys())
                print(f"Inner model forward parameters: {params}")

                # Check if position_ids is accepted
                if 'position_ids' in params:
                    print("  [OK] Model accepts 'position_ids' parameter")
                else:
                    print("  [WARNING] Model does NOT accept 'position_ids' parameter!")

                # Check if attention_mask is accepted
                if 'attention_mask' in params:
                    print("  [OK] Model accepts 'attention_mask' parameter")
                else:
                    print("  [WARNING] Model does NOT accept 'attention_mask' parameter!")
            except Exception as e:
                print(f"Could not inspect inner model signature: {e}")

        # Check TransformerLayer signature
        try:
            for name, module in model.named_modules():
                if 'TransformerLayer' in type(module).__name__:
                    sig = inspect.signature(module.forward)
                    params = list(sig.parameters.keys())
                    print(f"")
                    print(f"TransformerLayer ({name}) forward parameters: {params}")
                    break
        except Exception as e:
            print(f"Could not inspect TransformerLayer signature: {e}")

        # Check rotary embedding
        try:
            for name, module in model.named_modules():
                if 'Rotary' in type(module).__name__ or 'rotary' in name:
                    print(f"")
                    print(f"Rotary embedding module: {name}")
                    print(f"  Type: {type(module)}")
                    print(f"  Module: {module}")

                    try:
                        sig = inspect.signature(module.forward)
                        params = list(sig.parameters.keys())
                        print(f"  Forward parameters: {params}")
                    except Exception:
                        print("  Could not get forward signature")
                    break
        except Exception as e:
            print(f"Could not inspect rotary embedding: {e}")

        # Check DotProductAttention signature
        try:
            for name, module in model.named_modules():
                if 'DotProductAttention' in type(module).__name__:
                    sig = inspect.signature(module.forward)
                    params = list(sig.parameters.keys())
                    print(f"")
                    print(f"DotProductAttention forward parameters: {params}")
                    break
        except Exception as e:
            print(f"Could not inspect DotProductAttention signature: {e}")

    def _debug_test_forward_variants(self, input_ids, attention_mask, labels, position_ids):
        """
        DEBUG STEP 3: Test forward pass with different parameter combinations.
        """
        print("=" * 100)
        print("DEBUG STEP 3: FORWARD PASS VARIANTS TEST")
        print("=" * 100)

        seq_len = input_ids.shape[1]

        # Prepare default position_ids for comparison
        default_position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        # Prepare 2D attention mask (if we have 3D)
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask_2d = attention_mask.diagonal(dim1=-2, dim2=-1).long()
        elif attention_mask is not None:
            attention_mask_2d = attention_mask
        else:
            attention_mask_2d = None

        tests = [
            {
                "name": "A: No attention_mask, no position_ids (baseline)",
                "attention_mask": None,
                "position_ids": None,
            },
            {
                "name": "B: 2D attention_mask only",
                "attention_mask": attention_mask_2d,
                "position_ids": None,
            },
            {
                "name": "C: Custom position_ids only",
                "attention_mask": None,
                "position_ids": position_ids,
            },
            {
                "name": "D: Default position_ids (continuous)",
                "attention_mask": None,
                "position_ids": default_position_ids,
            },
            {
                "name": "E: 2D attention_mask + custom position_ids",
                "attention_mask": attention_mask_2d,
                "position_ids": position_ids,
            },
        ]

        # Only test 3D mask if it exists and is small enough
        if attention_mask is not None and attention_mask.dim() == 3:
            tests.append({
                "name": "F: 3D attention_mask + custom position_ids",
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            })

        results = []

        for test in tests:
            test_name = test["name"]
            try:
                with torch.no_grad():
                    forward_kwargs = {
                        "input_ids": input_ids,
                        "labels": labels,
                    }
                    if test["attention_mask"] is not None:
                        forward_kwargs["attention_mask"] = test["attention_mask"]
                    if test["position_ids"] is not None:
                        forward_kwargs["position_ids"] = test["position_ids"]

                    output = self.model(**forward_kwargs)
                    loss = output["loss"].item() if isinstance(output, dict) else output.loss.item()

                results.append((test_name, loss, "OK"))
                print(f"  {test_name}: loss = {loss:.4f}")

            except Exception as e:
                results.append((test_name, None, str(e)[:100]))
                print(f"  {test_name}: FAILED - {str(e)[:100]}")

        # Analysis
        print("")
        print("  Analysis:")

        # Compare C vs D (custom vs default position_ids)
        loss_c = next((r[1] for r in results if "C:" in r[0] and r[1] is not None), None)
        loss_d = next((r[1] for r in results if "D:" in r[0] and r[1] is not None), None)

        if loss_c is not None and loss_d is not None:
            diff = abs(loss_c - loss_d)
            if diff < 0.001:
                print(f"  [PROBLEM] Custom position_ids (C) vs default (D) have same loss!")
                print(f"            This suggests position_ids are being IGNORED by the model.")
            else:
                print(f"  [OK] Custom position_ids affect loss (diff = {diff:.4f})")

        # Compare A vs B (with/without attention mask)
        loss_a = next((r[1] for r in results if "A:" in r[0] and r[1] is not None), None)
        loss_b = next((r[1] for r in results if "B:" in r[0] and r[1] is not None), None)

        if loss_a is not None and loss_b is not None:
            diff = abs(loss_a - loss_b)
            if diff < 0.001:
                print(f"  [INFO] 2D attention_mask has minimal effect (diff = {diff:.4f})")
            else:
                print(f"  [INFO] 2D attention_mask affects loss (diff = {diff:.4f})")

        # Check if 3D mask works
        loss_f = next((r[1] for r in results if "F:" in r[0] and r[1] is not None), None)
        if loss_f is not None and loss_c is not None:
            diff = abs(loss_f - loss_c)
            print(f"  [INFO] 3D mask vs no mask diff = {diff:.4f}")

    def _debug_verify_rope_source(self):
        """
        DEBUG STEP 4: Check if RoPE implementation uses position_ids.
        """
        print("=" * 100)
        print("DEBUG STEP 4: ROPE SOURCE CODE VERIFICATION")
        print("=" * 100)

        # Find rotary embedding module
        rotary_module = None
        rotary_name = None
        for name, module in self.model.named_modules():
            if 'rotary' in name.lower() or 'Rotary' in type(module).__name__:
                rotary_module = module
                rotary_name = name
                break

        if rotary_module is None:
            print("  Could not find rotary embedding module")
            return

        print(f"  Found rotary module: {rotary_name}")
        print(f"  Type: {type(rotary_module)}")

        # Check source code if possible
        try:
            source = inspect.getsource(type(rotary_module).forward)

            # Check if position_ids is used
            if 'position_ids' in source or 'pos_ids' in source or 'seq_idx' in source:
                print("  [OK] Rotary forward method references position_ids or similar")
            else:
                print("  [WARNING] Rotary forward method does NOT reference position_ids!")
                print("  This means custom position_ids will have NO effect!")

            # Log first 1000 chars of source
            print(f"  Source preview (first 1000 chars):")
            for line in source[:1000].split('\n'):
                print(f"    {line}")

        except Exception as e:
            print(f"  Could not inspect rotary source: {e}")

    # =========================================================================
    # TRAINING STEP METHODS
    # =========================================================================

    def _training_step_dpo(self, batch, batch_idx, *a, **kw):
        prompt_ids, prompt_mask, chosen_ids, chosen_mask, rejected_ids, rejected_mask = self._prepare_dpo_input_batch(
            batch, batch_idx
        )
        dpo_params = {
            "model": self,
            "ref_model": self.ref_model,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
            "max_length": self._cfg.max_context_width,
            "beta": self._cfg.dpo.get("beta", 0.1),
            "label_smoothing": self._cfg.dpo.get("label_smoothing", 0.0),
            "peft": self.use_peft,
            "bf16": self._cfg.precision == "bf16",
        }
        if self.use_smp_model and self._cfg.fp8:
            fp8 = self._cfg.fp8
            fp8_recipe = self.fp8_recipe
            fp8_group = tsm.state.world_process_group
            with transformer_engine.pytorch.fp8_autocast(
                enabled=fp8,
                fp8_recipe=fp8_recipe,
                fp8_group=fp8_group,
            ):
                loss = compute_dpo_loss(
                    *a,
                    **dpo_params,
                    **kw,
                )
        else:
            loss = compute_dpo_loss(
                *a,
                **dpo_params,
                **kw,
            )
        return loss

    def _training_step_fp8(self, batch, batch_idx, *a, **kw):
        fp8 = self._cfg.fp8
        fp8_recipe = self.fp8_recipe
        fp8_group = tsm.state.world_process_group

        input_ids, attention_mask, labels, position_ids = self._prepare_input_batch(batch, batch_idx)

        # Check if sequence packing is enabled
        use_packing = os.environ.get("USE_SEQUENCE_PACKING", "false").lower() == "true"

        # =========================================================================
        # DEBUG: Log batch structure for first few batches
        # =========================================================================
        if batch_idx < 3 and dist.get_rank() == 0:
            self._debug_log_batch_structure(
                batch_idx=batch_idx,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
                use_packing=use_packing
            )

        # =========================================================================
        # DEBUG: Log model signature on first batch
        # =========================================================================
        if batch_idx == 0 and dist.get_rank() == 0:
            self._debug_log_model_signature()
            self._debug_verify_rope_source()

        # =========================================================================
        # DEBUG: Test forward variants on first batch
        # =========================================================================
        if batch_idx == 0 and dist.get_rank() == 0:
            self._debug_test_forward_variants(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids
            )

        with transformer_engine.pytorch.fp8_autocast(
            enabled=fp8,
            fp8_recipe=fp8_recipe,
            fp8_group=fp8_group,
        ):
            forward_kwargs = {
                "input_ids": input_ids,
                "labels": labels,
            }

            # Only add optional parameters if they exist
            if use_packing:
                if attention_mask is not None:
                    forward_kwargs["attention_mask"] = attention_mask
                if position_ids is not None:
                    forward_kwargs["position_ids"] = position_ids
            else:
                # Standard mode: attention_mask=None (default causal)
                forward_kwargs["attention_mask"] = None

            return self(*a, **forward_kwargs, **kw)["loss"]

    def _training_step(self, batch, batch_idx, *a, **kw):
        if self._cfg.get("multi_modal", None):
            return self(*a, **batch, **kw)["loss"]

        input_ids, attention_mask, labels, position_ids = self._prepare_input_batch(batch, batch_idx)

        # Check if sequence packing is enabled
        use_packing = os.environ.get("USE_SEQUENCE_PACKING", "false").lower() == "true"

        # =========================================================================
        # DEBUG: Log batch structure for first few batches
        # =========================================================================
        if batch_idx < 3 and dist.get_rank() == 0:
            self._debug_log_batch_structure(
                batch_idx=batch_idx,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
                use_packing=use_packing
            )

        # =========================================================================
        # DEBUG: Log model signature on first batch
        # =========================================================================
        if batch_idx == 0 and dist.get_rank() == 0:
            self._debug_log_model_signature()
            self._debug_verify_rope_source()

        # =========================================================================
        # DEBUG: Test forward variants on first batch
        # =========================================================================
        if batch_idx == 0 and dist.get_rank() == 0:
            self._debug_test_forward_variants(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids
            )

        forward_kwargs = {
            "input_ids": input_ids,
            "labels": labels,
        }

        # Only add optional parameters if they exist
        if use_packing:
            if attention_mask is not None:
                forward_kwargs["attention_mask"] = attention_mask
            if position_ids is not None:
                forward_kwargs["position_ids"] = position_ids
        else:
            # Standard mode: attention_mask=None (default causal)
            forward_kwargs["attention_mask"] = None

        return self(*a, **forward_kwargs, **kw)["loss"]

    def training_step(self, batch, batch_idx, *a, **kw):
        """
        General training forward steps, backward/optimizer step will be done by
        PTL can also skip auto optimization with self.automatic_optimization=False
        """
        if self._cfg.dpo.get("enabled", False):
            self.loss = self._training_step_dpo(batch, batch_idx, *a, **kw)
        elif self.use_smp_model and self._cfg.fp8:
            self.loss = self._training_step_fp8(batch, batch_idx, *a, **kw)
        else:
            self.loss = self._training_step(batch, batch_idx, *a, **kw)
        return self.loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        if self._cfg.get("dpo", False):
            prompt_ids, prompt_mask, chosen_ids, chosen_mask, rejected_ids, rejected_mask = (
                self._prepare_dpo_input_batch(batch, batch_idx)
            )
            dpo_params = {
                "model": self,
                "ref_model": self.ref_model,
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "chosen_ids": chosen_ids,
                "chosen_mask": chosen_mask,
                "rejected_ids": rejected_ids,
                "rejected_mask": rejected_mask,
                "max_length": self._cfg.max_context_width,
                "beta": self._cfg.dpo.get("beta", 0.1),
                "label_smoothing": self._cfg.dpo.get("label_smoothing", 0.0),
                "peft": self.use_peft,
                "bf16": self._cfg.precision == "bf16",
            }
            val_loss = compute_dpo_loss(**dpo_params)
        else:
            input_ids, attention_mask, labels, position_ids = self._prepare_input_batch(batch, batch_idx)

            # Check if sequence packing is enabled
            use_packing = os.environ.get("USE_SEQUENCE_PACKING", "false").lower() == "true"

            forward_kwargs = {
                "input_ids": input_ids,
                "labels": labels,
            }

            # Only add optional parameters if they exist
            if use_packing:
                if attention_mask is not None:
                    forward_kwargs["attention_mask"] = attention_mask
                if position_ids is not None:
                    forward_kwargs["position_ids"] = position_ids
            else:
                # Standard mode: attention_mask=None (default causal)
                forward_kwargs["attention_mask"] = None

            val_loss = self(**forward_kwargs)["loss"]

        self.val_loss += val_loss.detach()
        return val_loss

    def _prepare_dpo_input_batch(self, batch, batch_idx):
        """
        Parse input batch, pre-process for DPO context parallel
        """
        prompt_ids, prompt_mask, chosen_ids, chosen_mask, rejected_ids, rejected_mask = (
            self.trainer.datamodule.get_batch(batch)
        )
        self.batch_num_sequences = prompt_ids.shape[0]
        if self._cfg.get("context_parallel_degree", 1) > 1:
            # Apply context parallel processing to all tensors
            prompt_ids, prompt_mask, chosen_ids, chosen_mask, rejected_ids, rejected_mask = get_batch_for_cp_rank(
                (prompt_ids, prompt_mask, chosen_ids, chosen_mask, rejected_ids, rejected_mask)
            )
        if batch_idx == 0:
            # checking only on batch 0 to reduce checks during runtime
            chosen_width = prompt_ids.shape[1] + chosen_ids.shape[1]
            rejected_width = prompt_ids.shape[1] + rejected_ids.shape[1]
            width_over_degree = self._cfg.max_context_width // self._cfg.get("context_parallel_degree", 1)
            if chosen_width != width_over_degree or rejected_width != width_over_degree:
                _logger.warning(
                    f"Warning: input data passed {prompt_ids.shape}, {chosen_ids.shape}, {rejected_ids.shape} does not respect max_context_width set. If context parallelism is enabled,",
                    f"Completion input_ids sequence length == (model.max_context_width / model.context_parallel_degree) ",
                )
        return prompt_ids, prompt_mask, chosen_ids, chosen_mask, rejected_ids, rejected_mask

    def _prepare_input_batch(self, batch, batch_idx):
        """
        Parse input batch, pre-process for context parallel
        Supports both:
        - Standard mode: (input_ids, attention_mask, labels)
        - Sequence packing mode: (input_ids, attention_mask, labels, position_ids)
        """
        # Check if sequence packing is enabled
        use_packing = os.environ.get("USE_SEQUENCE_PACKING", "false").lower() == "true"

        batch_data = self.trainer.datamodule.get_batch(batch)

        if use_packing and len(batch_data) == 4:
            # Sequence packing mode: extract all 4 components
            input_ids, attention_mask, labels, position_ids = batch_data
        elif len(batch_data) == 3:
            # Standard mode: only 3 components
            input_ids, attention_mask, labels = batch_data
            position_ids = None
        else:
            # Fallback for legacy format where attention_mask might be ignored
            input_ids, _, labels = batch_data
            attention_mask = None
            position_ids = None

        self.batch_num_sequences = input_ids.shape[0]

        if self._cfg.get("context_parallel_degree", 1) > 1:
            if use_packing and position_ids is not None:
                # Pack all components for context parallel
                input_ids, attention_mask, labels, position_ids = get_batch_for_cp_rank(
                    (input_ids, attention_mask, labels, position_ids)
                )
            elif attention_mask is not None:
                # Pack with attention mask but no position_ids
                input_ids, attention_mask, labels = get_batch_for_cp_rank(
                    (input_ids, attention_mask, labels)
                )
            else:
                # Legacy: only input_ids and labels
                input_ids, labels = get_batch_for_cp_rank((input_ids, labels))

        if batch_idx == 0 and dist.get_rank() == 0:
            # checking only on batch 0 to reduce checks during runtime
            if (self._cfg.get("context_parallel_degree", 1) > 1) & (
                input_ids.shape[1] != (self._cfg.max_context_width // self._cfg.get("context_parallel_degree", 1))
            ):
                _logger.warning(
                    f"Input data passed {input_ids.shape} does not respect max_context_width set. "
                    f"If context parallelism is enabled, input_ids sequence length should be "
                    f"(model.max_context_width / model.context_parallel_degree)."
                )

        return input_ids, attention_mask, labels, position_ids

    def _compute_packed_sequence_loss(self, logits, labels):
        """
        Custom loss for packed sequences
        """
        shift_logits = logits[..., :-1, :].contiguous()  # Remove last token
        shift_labels = labels[..., 1:].contiguous()  # Remove first token

        # Get EOS token ID
        if hasattr(self.trainer.datamodule, 'tokenizer') and self.trainer.datamodule.tokenizer is not None:
            eos_token_id = self.trainer.datamodule.tokenizer.eos_token_id
        else:
            # Default for Llama models
            eos_token_id = 2
        loss_mask = (shift_labels != eos_token_id).float()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # Flatten
            shift_labels.view(-1)
        )
        masked_loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()
        return masked_loss

    def setup_optimization(
        self,
        optim_config: Optional[Union[DictConfig, Dict]] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        optim_config_cp = self._optim_config_copy(optim_config) or {}
        max_steps = optim_config_cp.get("sched", {}).get("max_steps")
        if "sched" in optim_config_cp and max_steps is None:
            with open_dict(optim_config_cp):
                optim_config_cp.sched.max_steps = self._get_max_steps()

        optim_kwargs = {} if optim_kwargs is None else optim_kwargs.copy()
        return super().setup_optimization(
            optim_config=optim_config_cp,
            optim_kwargs=optim_kwargs,
        )

    def configure_gradient_clipping(self, *args, **kwargs):
        grad_norm = clip_grad_norm_(self.model, self._cfg.grad_clip)
        self.grad_norm = grad_norm.detach().cpu()

    def configure_optimizers(self):
        self.setup_optimization()
        if getattr(self._cfg.optim, "sched", None) and self._scheduler is None:
            # The error below refers in particular to logs from
            # `prepare_lr_scheduler()` (when it retunrs `None`).
            raise AssertionError(
                "A scheduler config exists but no scheduler was instantiated!"
                "Previous logs may help identify the root cause of this issue."
            )
        if self._scheduler is None:
            return self._optimizer
        return [self._optimizer], [self._scheduler]

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def on_train_batch_start(self, *args, **kwargs):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, *args, **kwargs):
        """
        Hook called at the end of each training batch, do logging here
        """
        if self.trainer.strategy.cfg.log_perf_metrics:
            self.step_time = time.time() - self.batch_start_time
            self.sample_processed = self.batch_num_sequences * self.dp_size
            throughput = self.sample_processed / self.step_time

            tflops_gpu = compute_tflops(
                self.cfg, self.model_config, self.sample_processed, self.step_time, dist.get_world_size()
            )
            self.log("Step Time", self.step_time, prog_bar=True)
            self.log("TFLOPS/GPU", tflops_gpu, prog_bar=True)
            self.log("Samples/Second", throughput, prog_bar=True)

        loss_scalar = self._process_loss(self.loss, reduce_loss=self._cfg.log_reduced_training_loss)
        self.log("Loss/train", loss_scalar, prog_bar=True)
        self.log("Norms/grad_norm", self.grad_norm, prog_bar=True)
        self.log("LR/learning_rate", self.lr_schedulers().get_lr()[0], prog_bar=True)

    def on_validation_epoch_end(self):
        """
        Hook called at the end of each validation epoch, do logging here
        """
        if self.trainer.global_step > 0:
            loss_scalar = self._process_loss(self.val_loss)
            loss_scalar /= self.trainer.num_val_batches[0]
            ppl = math.exp(loss_scalar)
            if dist.get_rank() == 0:
                _logger.info(
                    f"Done validation after step {self.global_step}, validation loss: {loss_scalar}, validation perplexity: {ppl}"
                )
            self.log("Loss/val", loss_scalar)
            self.log("Loss/perplexity", ppl)
            self.val_loss = 0

    def _process_loss(self, loss, reduce_loss=True):
        """General function to process loss after train/eval"""
        if reduce_loss:
            loss_detached = loss.detach()
            dist.all_reduce(loss_detached)
            loss_scalar = loss_detached.item() / dist.get_world_size()
            return loss_scalar

        loss_scalar = self.loss.item()
        return loss_scalar

    def _get_max_steps(self):
        """
        Compute the maximum number of training steps (-1 if it cannot be computed).
        Over write nemo's _get_max_steps with
        1. Override max step from config lr_decay_iters
        2. Get data loader length from datamodule
        """
        if self._cfg.lr_decay_iters is not None:
            return self._cfg.lr_decay_iters

        if getattr(self, "_trainer", None) is None:
            _logger.warning("Cannot compute `max_steps` as no trainer is set")
            return -1

        if self._trainer.max_steps >= 0:
            # Note that when `trainer.max_steps` is defined, we ignore
            # `max_epochs` (even if training may end before `max_steps` is
            # reached due to `max_epochs`). This is for backward compatibility
            # with older versions of NeMo.
            if self._trainer.max_epochs is not None and self._trainer.max_epochs >= 0:
                _logger.warning(
                    "Ignoring `trainer.max_epochs` when computing `max_steps` "
                    "because `trainer.max_steps` is already "
                    f"set to {self._trainer.max_steps}."
                )
            return self._trainer.max_steps

        if self._trainer.max_epochs is None or self._trainer.max_epochs < 0:
            _logger.warning("Cannot compute `max_steps` if neither `trainer.max_steps` nor `trainer.max_epochs` is set")
            return -1

        if getattr(self, "_train_dl", None) is None:
            _logger.warning("Cannot compute `max_steps` from the number of epochs as the train dataloader is not set")
            return -1

        # The number of training step per epoch is typically the number of
        # global batches in the training set...
        num_global_batches = len(self.datamodule._train_dl)
        steps_per_epoch = num_global_batches

        # ... unless it is constrained by the `limit_train_batches` option.
        limit_batches = self._trainer.limit_train_batches
        if limit_batches is not None:
            if isinstance(limit_batches, float):
                limit_batches = int(limit_batches * num_global_batches)

            steps_per_epoch = min(num_global_batches, limit_batches)

        return steps_per_epoch * self._trainer.max_epochs

    def list_available_models(self):
        """Override Nemo's abstract class"""
        return None

    def setup_training_data(self):
        """We're using Data Module for data pipelining"""
        return None

    def setup_validation_data(self):
        """We're using Data Module for data pipelining"""
        return None