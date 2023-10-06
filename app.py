
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
#model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')
# model_name = "gpt2-large"  # You can use "gpt2-medium", "gpt2-large", etc., for larger models
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

'''
The following is llama2 version
'''
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",load_in_4bit=True)

import copy
# Split function of llama2
def split_v2(model):
    llama2Layers = []
    part1 = []
    part2 = []
    part3 = []
    
    part1.append(copy.deepcopy(model.model.embed_tokens))
    
    part2.append(copy.deepcopy(model.model.layers))
    part2.append(copy.deepcopy(model.model.norm))
    
    part3.append(model.lm_head)
    
    return part1, part2, part3

part1, part2, part3 = split_v2(model)

from typing import Optional, List, Union, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import BaseModelOutputWithPast

def forward(
        part1,
        part2,
        config,
        info,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else config.use_cache

        return_dict = return_dict if return_dict is not None else config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # Get the embed_tokens from part 1 instead of llama_model
        embed_tokens = part1[0]
        if inputs_embeds is None:
            inputs_embeds = embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if info.gradient_checkpointing and info.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Get the layers from part 2 instead of llama_model
        layers = part2[0]
        for idx, decoder_layer in enumerate(layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if info.gradient_checkpointing and info.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Get the norm from part 2 instead of llama_model
        norm = part2[1]
        hidden_states = norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

def myForward_v3(
        part1,
        part2,
        part3,
        
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    )-> Union[Tuple, CausalLMOutputWithPast]:
    llama_model = model.model
    
    
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

    lm_head = part3[0]
    
    # Get the information of llama model
    config = llama_model.config
    info = Info()
    info.gradient_checkpointing = llama_model.gradient_checkpointing
    info.training = llama_model.training
    
    # outputs = forward(
    #     part1,
    #     part2,
    #     config,
    #     info,
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     position_ids=position_ids,
    #     past_key_values=past_key_values,
    #     inputs_embeds=inputs_embeds,
    #     use_cache=use_cache,
    #     output_attentions=output_attentions,
    #     output_hidden_states=output_hidden_states,
    #     return_dict=return_dict,
    # )
    
    hidden_states, position_ids, attention_mask =  cal1(
        part1,
        config,
        info,
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
    outputs = cal2(
        part2,
        config,
        info,
        hidden_states = hidden_states,
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
    
    
    # The default config.pretraining_tp is 1 for faster computation
    logits = lm_head(hidden_states)
    logits = logits.float()
    
    loss = None
    
    return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def cal1(
        part1,
        config,
        info,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:


        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # Get the embed_tokens from part 1 instead of llama_model
        embed_tokens = part1[0]
        if inputs_embeds is None:
            inputs_embeds = embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        
        return hidden_states, position_ids, attention_mask


def cal2(
        part2,
        config,
        info,
        hidden_states,
        position_ids, 
        attention_mask,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        use_cache = use_cache if use_cache is not None else config.use_cache

        return_dict = return_dict if return_dict is not None else config.use_return_dict

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else config.output_hidden_states
        )

        output_attentions = output_attentions if output_attentions is not None else config.output_attentions

        
        if info.gradient_checkpointing and info.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Get the layers from part 2 instead of llama_model
        layers = part2[0]
        for idx, decoder_layer in enumerate(layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if info.gradient_checkpointing and info.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Get the norm from part 2 instead of llama_model
        norm = part2[1]
        hidden_states = norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def myGenerate_v3(part1, part2,part3, input, max_length = 20):
  input_ids = tokenizer.encode("<startofstring> "+input+" <bot>:", return_tensors="pt").to(device)
  # Set the maximum length for the generated sequence
  # Disable gradient calculation for faster inference
  with torch.no_grad():
    # Forward pass to get initial predictions
    output = myForward_v3(part1, part2,part3, input_ids)
    
    # Get the last predicted token
    last_token_logits = output.logits[:, -1, :]
    
    # Set the temperature for sampling (higher values make output more random)
    temperature = 0.1
    # Apply temperature to logits for sampling
    logits = last_token_logits / temperature
    # Generate the next token by sampling from the logits
    next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

    # Append the generated token to the input sequence
    input_ids = torch.cat([input_ids, next_token], dim=-1)
    # Generate the remaining tokens of the sequence
    for _ in range(max_length - 1):
        # Forward pass with the updated input sequence
        output = myForward_v3(part1,part2,part3, input_ids)
        
        # Get the last predicted token
        last_token_logits = output.logits[:, -1, :]

        # Apply temperature to logits for sampling
        logits = last_token_logits / temperature

        # Generate the next token by sampling from the logits
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

        # Append the generated token to the input sequence
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    # Convert the generated sequence tensor to text
    output_text = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)

  return output_text


# Usage
# text = "Hey, are you conscious? Can you talk to me"
# inputs = tokenizer(text, return_tensors="pt").to(device)
# outputs = myGenerate_v3(part1, part2, part3, text, 30)
# print(outputs)

app = Flask(__name__)

# initiate input
step = 0
chat_history_ids = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    global step
    global chat_history_ids
    question = request.form['user_input']
    
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    #new_user_input_ids = tokenizer.encode(question + tokenizer.eos_token, return_tensors='pt')

    # add new input tokens to chat history
    #bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    # input_ids = tokenizer.encode(question, return_tensors="pt")

    # generate response
    # response_ids = model.generate(
    #     input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id
    # )

    
    #answer = ("{}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
    # answer = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    answer = myGenerate_v3(part1, part2, part3, question, 30)
    step += 1

    return answer

if __name__ == '__main__':
    app.run(debug=True)
