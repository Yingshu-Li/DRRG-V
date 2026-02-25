#    Copyright 2023 Zebin You
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
import torch.distributed as dist

from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.configuration_qwen import Qwen3Config
from llava.model.language_model.modeling_Qwem import Qwen3Model, Qwen3ForCausalLM


class LLavaQwen3Config(Qwen3Config):
    model_type = "llava_qwen3"      
    temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None
    # rope_scaling: Optional[dict] = {}


class LlavaQwen3Model(LlavaMetaModel, Qwen3Model):
    config_class = LLavaQwen3Config

    def __init__(self, config: Qwen3Config):
        super(LlavaQwen3Model, self).__init__(config)

class LlavaQwen3ModelLM(Qwen3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LLavaQwen3Config

    def __init__(self, config):
        Qwen3ForCausalLM.__init__(self, config)

        # configure default generation settings
        config.model_type = "llava_qwen3"
        # config.rope_scaling = None

        self.model = LlavaQwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.chexbert_head = nn.Linear(1152, 14, bias=False)
        self.concept_table = self.concept_table = nn.Parameter(torch.empty(14, 1152))
        nn.init.trunc_normal_(self.concept_table, mean=0.0, std=0.02, a=-0.04, b=0.04)
        self.fuse_head = nn.Linear(1152, 1152, bias=False) 
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position=None,
        sampled_finding_token_ids: Optional[list] = None,
        remaining_finding_token_ids: Optional[list] = None,
        all_finding_token_ids: Optional[list] = None,
        chexbert_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #Print batch size and sequence length for debugging
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            bs = input_ids.shape[0] if input_ids is not None else (inputs_embeds.shape[0] if inputs_embeds is not None else None)
            seqlen = input_ids.shape[1] if input_ids is not None else (inputs_embeds.shape[1] if inputs_embeds is not None else None)
            print(f"[DEBUG] model.forward bs={bs}, seqlen={seqlen}, has_images={images is not None}")
        if inputs_embeds is None and attention_mask is not None:
            # donate multi-dialogue 
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, conversation_ids, concept_loss) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, is_llada=True, chexbert_labels=chexbert_labels)
        elif inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, concept_loss) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, chexbert_labels=chexbert_labels)
            conversation_ids = None
        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                concept_loss=concept_loss,
                conversation_ids=conversation_ids,
                sampled_finding_token_ids=sampled_finding_token_ids,
                remaining_finding_token_ids=remaining_finding_token_ids,
                all_finding_token_ids=all_finding_token_ids,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, _,) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate_with_embeds(inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen3", LLavaQwen3Config)
AutoModelForCausalLM.register(LLavaQwen3Config, LlavaQwen3ModelLM)