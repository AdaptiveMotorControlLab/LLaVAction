#    Copyright 2024 Hao Zhang
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
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llavaction.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig




class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)      
        Qwen2ForCausalLM.__init__(self, config)
        
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing

        if getattr(self.config, "vision_supervision", None) is not None:
            assert config.action_types is not None
            action_types = [int(x) for x in config.action_types.split(",")]
            self.verb_head = nn.Linear(config.hidden_size, action_types[0], bias=False)
            self.noun_head = nn.Linear(config.hidden_size, action_types[1], bias=False)
            self.action_head = nn.Linear(config.hidden_size, action_types[2], bias=False)


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
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        actions: Optional[torch.LongTensor] = None,
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, action_idx) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

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
            # return super().forward(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     past_key_values=past_key_values,
            #     inputs_embeds=inputs_embeds,
            #     labels=labels,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            all_states = None
            if len(outputs) > 1:
                all_states = outputs[1]
            
            logits = self.lm_head(hidden_states)
            logits = logits.float()
            
            if getattr(self.config, "vision_supervision", None) is not None and actions is not None:
                # move the action_idx to the device of the hidden_states
                vision_supervision = self.config.vision_supervision
                action_idx = action_idx.to(hidden_states.device)
                action_states = hidden_states[torch.arange(hidden_states.size(0)).unsqueeze(1), action_idx, :]
                if vision_supervision in ["newline", "one_token"]:
                    verb_logits = self.verb_head(action_states[:, 0])
                    noun_logits = self.noun_head(action_states[:, 0])
                    action_logits = self.action_head(action_states[:, 0])
                    
                    # get those logits also for other layers
                    get_visual_tokens = lambda x: x[torch.arange(hidden_states.size(0)).unsqueeze(1), action_idx, :]
                    device = action_states.device
                    
                    other_layers = [get_visual_tokens(layer.to(device)) for layer in all_states]  # Move all layers at once
                    
                    other_verb_logits_list = []
                    other_noun_logits_list = []
                    other_action_logits_list = []
                    
                    for other_layer in other_layers:
                        other_verb_logits = self.verb_head(other_layer[:,0])
                        other_noun_logits = self.noun_head(other_layer[:,0])
                        other_action_logits = self.action_head(other_layer[:,0])
                        other_verb_logits_list.append(other_verb_logits)
                        other_noun_logits_list.append(other_noun_logits)
                        other_action_logits_list.append(other_action_logits)
                                         
                    
                elif vision_supervision == "all_newlines":
                    # note get logits for all new lines
                    verb_logits = self.verb_head(action_states)
                    noun_logits = self.noun_head(action_states) 
                    action_logits = self.action_head(action_states) 

                elif vision_supervision == "three_tokens":
                    verb_logits = self.verb_head(action_states[:, 0])
                    noun_logits = self.noun_head(action_states[:, 1])
                    action_logits = self.action_head(action_states[:, 2])
                     
                    get_visual_tokens = lambda x: x[torch.arange(hidden_states.size(0)).unsqueeze(1), action_idx,:]
                    device = action_states.device
                    
                    other_layers = [get_visual_tokens(layer.to(device)) for layer in all_states]  # Move all layers at once
                    
                    other_verb_logits_list = []
                    other_noun_logits_list = []
                    other_action_logits_list = []
                    
                    for other_layer in other_layers:
                        other_verb_logits = self.verb_head(other_layer[:,0])
                        other_noun_logits = self.noun_head(other_layer[:,1])
                        other_action_logits = self.action_head(other_layer[:,2])
                        other_verb_logits_list.append(other_verb_logits)
                        other_noun_logits_list.append(other_noun_logits)
                        other_action_logits_list.append(other_action_logits)                    
                
            loss = None
            if labels is not None:          
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

                if getattr(self.config, "vision_supervision", None) is not None and actions is not None:
                    device = shift_logits.device
                 
                    verb_logits = verb_logits.to(device)
                    noun_logits = noun_logits.to(device)
                    action_logits = action_logits.to(device)
                    actions = actions.to(device)
                                      
                    vision_supervision_loss = 0.0                    
                    
                    triples = list(zip(other_verb_logits_list, other_noun_logits_list, other_action_logits_list))
                                      
                    if getattr(self.config, 'vision_token_training', None) and self.config.vision_token_training == 'last_layer':
                        triples = triples[-1:]
                    elif getattr(self.config, 'vision_token_training', None) and self.config.vision_token_training == 'first_layer':
                        triples = triples[:1]
                    elif getattr(self.config, 'vision_token_training', None) and self.config.vision_token_training == 'all_layers':
                        pass
                    # by default, distilaltion uses all layers            
                    # First check if any process has valid examples across all triples
                    world_has_valid = torch.tensor(actions[:, 0].any() >= 0, device=actions.device)
                    torch.distributed.all_reduce(world_has_valid, op=torch.distributed.ReduceOp.MAX)               

                    if world_has_valid:  # If any process has valid examples
                        for other_verb_logits, other_noun_logits, other_action_logits in triples:
                            valid_mask = actions[:, 0] >= 0
                            
                            if valid_mask.any():  # This process has valid examples
                                valid_verb_logits = other_verb_logits[valid_mask]
                                valid_noun_logits = other_noun_logits[valid_mask]
                                valid_action_logits = other_action_logits[valid_mask]
                                
                                valid_verb_targets = actions[valid_mask, 0]
                                valid_noun_targets = actions[valid_mask, 1]
                                valid_action_targets = actions[valid_mask, 2]

                                other_verb_loss = loss_fct(valid_verb_logits, valid_verb_targets)
                                other_noun_loss = loss_fct(valid_noun_logits, valid_noun_targets)
                                other_action_loss = loss_fct(valid_action_logits, valid_action_targets)
                                
                                vision_supervision_loss += 0.5 * other_verb_loss + 0.5 * other_noun_loss + 0.1 * other_action_loss
                            else:  # This process has no valid examples but others do
                                # Add dummy loss to maintain gradient flow
                                vision_supervision_loss += 0.0 * (other_verb_logits.sum() + other_noun_logits.sum() + other_action_logits.sum())

                        vision_supervision_loss /= (len(triples) + 1)
                        loss += vision_supervision_loss * 0.1
                    else:
                        # If no process has valid examples, add dummy loss to prevent hanging
                        dummy_loss = sum(sum(t.sum() * 0.0 for t in triple) for triple in triples)
                        vision_supervision_loss = dummy_loss / (len(triples) + 1)
                        loss += vision_supervision_loss * 0.1

                if getattr(self.config, 'vision_token_training', None) and  'distillation' in self.config.vision_token_training:
                    

                    distillation_loss = 0.0
                    if self.config.vision_token_training == 'normal_distillation':                    
                        teacher_layer_logits = get_visual_tokens(all_states[-1].to(device))
                    elif self.config.vision_token_training == 'reverse_distillation':
                        teacher_layer_logits = get_visual_tokens(all_states[0].to(device))
                        
                    teacher_layer_logits = teacher_layer_logits.contiguous().float().detach()
                    
                    if self.config.vision_token_training == 'normal_distillation':
                        student_layers = other_layers[:-1]
                    elif self.config.vision_token_training == 'reverse_distillation':
                        student_layers = other_layers[1:]
                    
                    for student_layer in student_layers:
                        teacher_layer_logits = teacher_layer_logits.contiguous().float().detach()
                        student_layer = student_layer.contiguous().float()
                        teacher_layer_probs = torch.nn.functional.softmax(teacher_layer_logits, dim=-1)
                        student_layer_log_probs = torch.nn.functional.log_softmax(student_layer, dim=-1)
                        # Compute KL divergence
                        distillation_loss += torch.nn.functional.kl_div(student_layer_log_probs, teacher_layer_probs, reduction="batchmean")
                        
                    distillation_loss /= len(other_layers)
                    loss += distillation_loss * 0.1

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
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
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
