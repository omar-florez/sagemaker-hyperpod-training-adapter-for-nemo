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
    
    Supports sequence packing via cu_seqlens injection into TransformerEngine DotProductAttention.
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

        # =========================================================================
        # SEQUENCE PACKING STATE (cu_seqlens approach)
        # =========================================================================
        self._cu_seqlens_q = None
        self._cu_seqlens_kv = None
        self._max_seqlen_q = None
        self._max_seqlen_kv = None
        self._packing_hooks = []
        
        # Check if sequence packing is enabled via config or environment
        use_packing_from_config = cfg.model.data.get("use_sequence_packing", None) if hasattr(cfg, 'model') and hasattr(cfg.model, 'data') else None
        use_packing_from_env = os.environ.get("USE_SEQUENCE_PACKING", "false").lower() == "true"
        self._use_packing = use_packing_from_config if use_packing_from_config is not None else use_packing_from_env
        
        if self._use_packing and dist.get_rank() == 0:
            _logger.info("=" * 80)
            _logger.info("SEQUENCE PACKING MODE ENABLED (cu_seqlens approach)")
            _logger.info("  cu_seqlens will be injected into DotProductAttention via hooks")
            _logger.info("=" * 80)
        # =========================================================================

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

    # =========================================================================
    # SEQUENCE PACKING: cu_seqlens HOOK METHODS
    # =========================================================================

    def _register_packing_hooks(self):
        """
        Register forward pre-hooks on all DotProductAttention modules
        to inject cu_seqlens for sequence packing.
        
        This enables TransformerEngine to properly isolate attention
        between packed documents without cross-document attention.
        """
        if not self._use_packing:
            return
        
        try:
            from transformer_engine.pytorch.attention import DotProductAttention
        except ImportError:
            _logger.warning("TransformerEngine not available, cannot register packing hooks")
            return
        
        hook_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, DotProductAttention):
                hook = module.register_forward_pre_hook(
                    self._dpa_pre_hook,
                    with_kwargs=True
                )
                self._packing_hooks.append(hook)
                hook_count += 1
        
        if dist.get_rank() == 0:
            _logger.info(f"Registered {hook_count} DotProductAttention hooks for cu_seqlens injection")
            if hook_count > 0:
                _logger.info("  Hook will inject: cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv")

    def _dpa_pre_hook(self, module, args, kwargs):
        """
        Forward pre-hook for DotProductAttention.
        
        Injects cu_seqlens parameters when sequence packing is active.
        This enables TransformerEngine to properly isolate attention
        between packed documents.
        
        Args:
            module: The DotProductAttention module
            args: Positional arguments to forward
            kwargs: Keyword arguments to forward
            
        Returns:
            Modified (args, kwargs) with cu_seqlens injected
        """
        if self._cu_seqlens_q is None:
            # No packing data for this batch, pass through unchanged
            return args, kwargs
        
        # Make a mutable copy of kwargs
        kwargs = dict(kwargs)
        
        # Inject cu_seqlens into kwargs
        kwargs['cu_seqlens_q'] = self._cu_seqlens_q
        kwargs['cu_seqlens_kv'] = self._cu_seqlens_kv
        kwargs['max_seqlen_q'] = self._max_seqlen_q
        kwargs['max_seqlen_kv'] = self._max_seqlen_kv
        
        return args, kwargs

    def _prepare_cu_seqlens_for_batch(self, cu_seqlens_list, max_seqlen, device):
        """
        Prepare cu_seqlens tensors for the current batch.
        
        For batch_size > 1, we concatenate each sample's cu_seqlens with
        appropriate offsets so TransformerEngine treats the flattened batch
        as a single sequence with multiple document boundaries.
        
        Args:
            cu_seqlens_list: List of cu_seqlens tensors, one per sample in batch
            max_seqlen: Maximum document length in batch
            device: Target device for tensors
        """
        if cu_seqlens_list is None:
            self._cu_seqlens_q = None
            self._cu_seqlens_kv = None
            self._max_seqlen_q = None
            self._max_seqlen_kv = None
            return
        
        batch_size = len(cu_seqlens_list)
        
        if batch_size == 1:
            # Simple case: single sample, use cu_seqlens directly
            cu_seqlens = cu_seqlens_list[0].to(device=device, dtype=torch.int32)
            self._cu_seqlens_q = cu_seqlens
            self._cu_seqlens_kv = cu_seqlens
            self._max_seqlen_q = max_seqlen
            self._max_seqlen_kv = max_seqlen
        else:
            # Multi-sample batch: Concatenate cu_seqlens with offsets
            # Each sample's cu_seqlens needs to be offset by the cumulative length
            # of previous samples
            
            # Get the padded sequence length from the first sample's final position
            # All samples should have the same padded length
            seq_len = cu_seqlens_list[0][-1].item()
            
            all_cu_seqlens = [torch.tensor([0], dtype=torch.int32, device=device)]
            offset = 0
            
            for b, sample_cu_seqlens in enumerate(cu_seqlens_list):
                # Move to device and add offset
                sample_cu_seqlens = sample_cu_seqlens.to(device=device, dtype=torch.int32)
                
                # Add this sample's boundaries (excluding initial 0) with offset
                boundaries = sample_cu_seqlens[1:] + offset
                all_cu_seqlens.append(boundaries)
                
                # Each sample contributes seq_len tokens to the flattened batch
                offset += seq_len
            
            cu_seqlens = torch.cat(all_cu_seqlens)
            
            self._cu_seqlens_q = cu_seqlens
            self._cu_seqlens_kv = cu_seqlens
            self._max_seqlen_q = max_seqlen
            self._max_seqlen_kv = max_seqlen
        
        # Debug logging (first few batches only)
        if not hasattr(self, '_cu_seqlens_log_count'):
            self._cu_seqlens_log_count = 0
        
        if self._cu_seqlens_log_count < 3 and dist.get_rank() == 0:
            _logger.info(f"[Batch {self._cu_seqlens_log_count}] cu_seqlens prepared:")
            _logger.info(f"  Shape: {self._cu_seqlens_q.shape}")
            _logger.info(f"  First 10 values: {self._cu_seqlens_q[:10].tolist()}")
            _logger.info(f"  Last 5 values: {self._cu_seqlens_q[-5:].tolist()}")
            _logger.info(f"  max_seqlen: {self._max_seqlen_q}")
            _logger.info(f"  num_documents: {len(self._cu_seqlens_q) - 1}")
            self._cu_seqlens_log_count += 1

    def _clear_cu_seqlens(self):
        """Clear cu_seqlens state after forward pass."""
        self._cu_seqlens_q = None
        self._cu_seqlens_kv = None
        self._max_seqlen_q = None
        self._max_seqlen_kv = None

    # =========================================================================
    # END SEQUENCE PACKING METHODS
    # =========================================================================

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
        
        # =========================================================================
        # SEQUENCE PACKING: Register hooks after model is ready
        # =========================================================================
        if self._use_packing:
            self._register_packing_hooks()
        # =========================================================================

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

    def _debug_log_batch_structure(self, batch_idx, input_ids, attention_mask, labels, cu_seqlens_list, max_seqlen):
        """
        DEBUG: Log complete batch structure to understand data flow for cu_seqlens mode.
        """
        print("=" * 100)
        print(f"DEBUG: BATCH STRUCTURE (batch_idx={batch_idx}) - cu_seqlens mode")
        print("=" * 100)

        batch_size, seq_len = input_ids.shape
        print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
        print(f"use_packing: {self._use_packing}")

        # Get tokenizer info
        try:
            tokenizer = self.trainer.datamodule.tokenizer
            eos_token_id = tokenizer.eos_token_id if tokenizer else 128001
            print(f"EOS token ID: {eos_token_id}")
        except Exception:
            eos_token_id = 128001
            print(f"EOS token ID (default): {eos_token_id}")

        IGNORE_INDEX = -100

        # Log cu_seqlens info
        if cu_seqlens_list is not None:
            print(f"\ncu_seqlens_list: {len(cu_seqlens_list)} samples")
            for i, cu_seqlens in enumerate(cu_seqlens_list[:2]):  # First 2 samples
                print(f"  Sample {i}: {cu_seqlens.shape}, values[:10]={cu_seqlens[:10].tolist()}")
                print(f"           num_docs={len(cu_seqlens)-1}, last_value={cu_seqlens[-1].item()}")
            print(f"max_seqlen: {max_seqlen}")
        else:
            print("\ncu_seqlens_list: None (standard mode)")

        for sample_idx in range(min(batch_size, 2)):  # Log first 2 samples
            print("-" * 80)
            print(f"SAMPLE {sample_idx}:")
            print("-" * 80)

            ids = input_ids[sample_idx].cpu()
            lab = labels[sample_idx].cpu()

            # Find EOS positions
            eos_positions = (ids == eos_token_id).nonzero(as_tuple=True)[0].tolist()
            print(f"  EOS positions: {eos_positions[:20]}{'...' if len(eos_positions) > 20 else ''}")
            print(f"  Number of EOS tokens: {len(eos_positions)}")

            # Find padding (attention_mask == 0)
            padding_start = seq_len
            if attention_mask is not None and attention_mask.dim() == 2:
                mask = attention_mask[sample_idx].cpu()
                padding_positions = (mask == 0).nonzero(as_tuple=True)[0]
                if len(padding_positions) > 0:
                    padding_start = padding_positions[0].item()

            print(f"  Padding starts at position: {padding_start}")
            print(f"  Real content length: {padding_start}")

            # Find IGNORE_INDEX positions in labels
            ignored_positions = (lab == IGNORE_INDEX).nonzero(as_tuple=True)[0].tolist()
            print(f"  Labels with IGNORE_INDEX (-100): {len(ignored_positions)} positions")

            # Verify cu_seqlens matches EOS positions
            if cu_seqlens_list is not None and sample_idx < len(cu_seqlens_list):
                cu_seqlens = cu_seqlens_list[sample_idx]
                # cu_seqlens should be [0, eos1+1, eos2+1, ...] 
                expected_boundaries = [0] + [pos + 1 for pos in eos_positions if pos + 1 <= padding_start]
                
                # Check if there's trailing content after last EOS
                if eos_positions and eos_positions[-1] + 1 < padding_start:
                    expected_boundaries.append(padding_start)
                
                actual_boundaries = cu_seqlens.tolist()
                
                print(f"\n  cu_seqlens verification:")
                print(f"    Expected (from EOS): {expected_boundaries[:10]}{'...' if len(expected_boundaries) > 10 else ''}")
                print(f"    Actual cu_seqlens:   {actual_boundaries[:10]}{'...' if len(actual_boundaries) > 10 else ''}")
                
                if expected_boundaries[:10] == actual_boundaries[:10]:
                    print(f"    [OK] cu_seqlens matches EOS positions")
                else:
                    print(f"    [WARNING] cu_seqlens does NOT match EOS positions!")

            # Show document structure
            print(f"\n  Document structure (first 5 docs):")
            if cu_seqlens_list is not None and sample_idx < len(cu_seqlens_list):
                cu_seqlens = cu_seqlens_list[sample_idx]
                for doc_idx in range(min(5, len(cu_seqlens) - 1)):
                    doc_start = cu_seqlens[doc_idx].item()
                    doc_end = cu_seqlens[doc_idx + 1].item()
                    doc_len = doc_end - doc_start
                    
                    # Count ignored labels in this document
                    doc_labels = lab[doc_start:doc_end].tolist()
                    ignored_in_doc = sum(1 for l in doc_labels if l == IGNORE_INDEX)
                    
                    print(f"    Doc {doc_idx}: positions [{doc_start}:{doc_end}] (len={doc_len}), ignored_labels={ignored_in_doc}")

            # Show token alignment around first few EOS positions
            print(f"\n  Token alignment around EOS positions:")
            print(f"  {'Pos':>5} | {'InputID':>8} | {'Label':>8} | Notes")
            print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*30}")

            positions_to_show = set()
            for eos_pos in eos_positions[:5]:
                positions_to_show.add(max(0, eos_pos - 1))
                positions_to_show.add(eos_pos)
                positions_to_show.add(min(seq_len - 1, eos_pos + 1))

            last_printed = -2
            for pos in sorted(positions_to_show):
                if pos >= seq_len:
                    continue
                if pos > last_printed + 1 and last_printed >= 0:
                    print(f"  {'...':>5} |")

                input_id = ids[pos].item()
                label = lab[pos].item()

                notes = []
                if input_id == eos_token_id:
                    notes.append("EOS")
                if label == IGNORE_INDEX:
                    notes.append("MASKED")
                if pos in eos_positions:
                    notes.append("<-- DOC BOUNDARY")

                notes_str = ", ".join(notes) if notes else ""
                print(f"  {pos:>5} | {input_id:>8} | {label:>8} | {notes_str}")
                last_printed = pos

    def _debug_log_model_signature(self):
        """
        DEBUG: Log model forward signature and DotProductAttention parameters.
        """
        print("=" * 100)
        print("DEBUG: MODEL FORWARD SIGNATURE")
        print("=" * 100)

        model = self.model
        print(f"Model type: {type(model)}")

        # Check DotProductAttention signature
        try:
            from transformer_engine.pytorch.attention import DotProductAttention
            
            for name, module in model.named_modules():
                if isinstance(module, DotProductAttention):
                    sig = inspect.signature(module.forward)
                    params = list(sig.parameters.keys())
                    print(f"\nDotProductAttention ({name}) forward parameters:")
                    print(f"  {params}")
                    
                    # Check for cu_seqlens parameters
                    if 'cu_seqlens_q' in params:
                        print("  [OK] DotProductAttention accepts 'cu_seqlens_q'")
                    else:
                        print("  [WARNING] DotProductAttention does NOT accept 'cu_seqlens_q'!")
                    
                    if 'cu_seqlens_kv' in params:
                        print("  [OK] DotProductAttention accepts 'cu_seqlens_kv'")
                    
                    if 'max_seqlen_q' in params:
                        print("  [OK] DotProductAttention accepts 'max_seqlen_q'")
                    
                    break
        except Exception as e:
            print(f"Could not inspect DotProductAttention: {e}")

        # Check hook registration
        print(f"\nSequence packing hooks registered: {len(self._packing_hooks)}")

    def _debug_verify_hooks_active(self):
        """
        DEBUG: Verify that hooks are properly injecting cu_seqlens.
        """
        print("=" * 100)
        print("DEBUG: HOOK VERIFICATION")
        print("=" * 100)

        print(f"_use_packing: {self._use_packing}")
        print(f"Number of registered hooks: {len(self._packing_hooks)}")
        print(f"_cu_seqlens_q is set: {self._cu_seqlens_q is not None}")
        
        if self._cu_seqlens_q is not None:
            print(f"  Shape: {self._cu_seqlens_q.shape}")
            print(f"  Device: {self._cu_seqlens_q.device}")
            print(f"  Dtype: {self._cu_seqlens_q.dtype}")
            print(f"  First 10 values: {self._cu_seqlens_q[:10].tolist()}")

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

        input_ids, attention_mask, labels, cu_seqlens_list, max_seqlen = self._prepare_input_batch(batch, batch_idx)

        # =========================================================================
        # DEBUG: Log batch structure for first few batches
        # =========================================================================
        if batch_idx < 3 and dist.get_rank() == 0:
            self._debug_log_batch_structure(
                batch_idx=batch_idx,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                cu_seqlens_list=cu_seqlens_list,
                max_seqlen=max_seqlen
            )

        # =========================================================================
        # DEBUG: Log model signature on first batch
        # =========================================================================
        if batch_idx == 0 and dist.get_rank() == 0:
            self._debug_log_model_signature()

        # =========================================================================
        # SEQUENCE PACKING: Prepare cu_seqlens for hooks
        # =========================================================================
        if self._use_packing and cu_seqlens_list is not None:
            self._prepare_cu_seqlens_for_batch(cu_seqlens_list, max_seqlen, input_ids.device)

        # =========================================================================
        # DEBUG: Verify hooks are active on first batch
        # =========================================================================
        if batch_idx == 0 and dist.get_rank() == 0:
            self._debug_verify_hooks_active()

        try:
            with transformer_engine.pytorch.fp8_autocast(
                enabled=fp8,
                fp8_recipe=fp8_recipe,
                fp8_group=fp8_group,
            ):
                forward_kwargs = {
                    "input_ids": input_ids,
                    "labels": labels,
                }

                # When using cu_seqlens, pass attention_mask=None to let cu_seqlens control attention
                if self._use_packing and cu_seqlens_list is not None:
                    forward_kwargs["attention_mask"] = None
                else:
                    # Standard mode: attention_mask=None (default causal)
                    forward_kwargs["attention_mask"] = None

                return self(*a, **forward_kwargs, **kw)["loss"]
        finally:
            # =========================================================================
            # SEQUENCE PACKING: Clear cu_seqlens state after forward pass
            # =========================================================================
            if self._use_packing:
                self._clear_cu_seqlens()

    def _training_step(self, batch, batch_idx, *a, **kw):
        if self._cfg.get("multi_modal", None):
            return self(*a, **batch, **kw)["loss"]

        input_ids, attention_mask, labels, cu_seqlens_list, max_seqlen = self._prepare_input_batch(batch, batch_idx)

        # =========================================================================
        # DEBUG: Log batch structure for first few batches
        # =========================================================================
        if batch_idx < 3 and dist.get_rank() == 0:
            self._debug_log_batch_structure(
                batch_idx=batch_idx,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                cu_seqlens_list=cu_seqlens_list,
                max_seqlen=max_seqlen
            )

        # =========================================================================
        # DEBUG: Log model signature on first batch
        # =========================================================================
        if batch_idx == 0 and dist.get_rank() == 0:
            self._debug_log_model_signature()

        # =========================================================================
        # SEQUENCE PACKING: Prepare cu_seqlens for hooks
        # =========================================================================
        if self._use_packing and cu_seqlens_list is not None:
            self._prepare_cu_seqlens_for_batch(cu_seqlens_list, max_seqlen, input_ids.device)

        # =========================================================================
        # DEBUG: Verify hooks are active on first batch
        # =========================================================================
        if batch_idx == 0 and dist.get_rank() == 0:
            self._debug_verify_hooks_active()

        try:
            forward_kwargs = {
                "input_ids": input_ids,
                "labels": labels,
            }

            # When using cu_seqlens, pass attention_mask=None to let cu_seqlens control attention
            if self._use_packing and cu_seqlens_list is not None:
                forward_kwargs["attention_mask"] = None
            else:
                # Standard mode: attention_mask=None (default causal)
                forward_kwargs["attention_mask"] = None

            return self(*a, **forward_kwargs, **kw)["loss"]
        finally:
            # =========================================================================
            # SEQUENCE PACKING: Clear cu_seqlens state after forward pass
            # =========================================================================
            if self._use_packing:
                self._clear_cu_seqlens()

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
        """Validation step with sequence packing support"""
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
            input_ids, attention_mask, labels, cu_seqlens_list, max_seqlen = self._prepare_input_batch(batch, batch_idx)

            # =========================================================================
            # SEQUENCE PACKING: Prepare cu_seqlens for hooks
            # =========================================================================
            if self._use_packing and cu_seqlens_list is not None:
                self._prepare_cu_seqlens_for_batch(cu_seqlens_list, max_seqlen, input_ids.device)

            try:
                forward_kwargs = {
                    "input_ids": input_ids,
                    "labels": labels,
                }

                # When using cu_seqlens, pass attention_mask=None to let cu_seqlens control attention
                if self._use_packing and cu_seqlens_list is not None:
                    forward_kwargs["attention_mask"] = None
                else:
                    forward_kwargs["attention_mask"] = None

                val_loss = self(**forward_kwargs)["loss"]
            finally:
                # =========================================================================
                # SEQUENCE PACKING: Clear cu_seqlens state after forward pass
                # =========================================================================
                if self._use_packing:
                    self._clear_cu_seqlens()

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
        Parse input batch, pre-process for context parallel.
        
        Supports both:
        - Standard mode: (input_ids, attention_mask, labels)
        - Sequence packing mode: (input_ids, attention_mask, labels, cu_seqlens_list, max_seqlen)
        
        Returns:
            Tuple of (input_ids, attention_mask, labels, cu_seqlens_list, max_seqlen)
            cu_seqlens_list and max_seqlen are None for standard (non-packed) batches.
        """
        batch_data = self.trainer.datamodule.get_batch(batch)

        if self._use_packing and len(batch_data) == 5:
            # Sequence packing mode: extract all 5 components
            input_ids, attention_mask, labels, cu_seqlens_list, max_seqlen = batch_data
        elif len(batch_data) == 3:
            # Standard mode: only 3 components
            input_ids, attention_mask, labels = batch_data
            cu_seqlens_list = None
            max_seqlen = None
        elif len(batch_data) == 4:
            # Legacy packing mode with position_ids (fallback)
            input_ids, attention_mask, labels, _ = batch_data
            cu_seqlens_list = None
            max_seqlen = None
            if dist.get_rank() == 0 and batch_idx == 0:
                _logger.warning("Received 4-tuple batch (legacy position_ids mode). cu_seqlens not available.")
        else:
            # Fallback for legacy format where attention_mask might be ignored
            input_ids, _, labels = batch_data[:3]
            attention_mask = None
            cu_seqlens_list = None
            max_seqlen = None

        self.batch_num_sequences = input_ids.shape[0]

        if self._cfg.get("context_parallel_degree", 1) > 1:
            # Note: cu_seqlens handling with context parallel requires careful consideration
            # For now, we only split input_ids, attention_mask, labels
            if attention_mask is not None:
                input_ids, attention_mask, labels = get_batch_for_cp_rank(
                    (input_ids, attention_mask, labels)
                )
            else:
                input_ids, labels = get_batch_for_cp_rank((input_ids, labels))
            
            # TODO: cu_seqlens would need to be adjusted for context parallel
            # For now, disable cu_seqlens with CP
            if cu_seqlens_list is not None and dist.get_rank() == 0 and batch_idx == 0:
                _logger.warning("cu_seqlens sequence packing with context_parallel_degree > 1 is not yet supported. Disabling cu_seqlens.")
                cu_seqlens_list = None
                max_seqlen = None

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

        return input_ids, attention_mask, labels, cu_seqlens_list, max_seqlen

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