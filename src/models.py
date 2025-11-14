import os
import gc
import torch
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import conversation_temp


PATH_DICT = {
    "llama-3": "/home/cshdtian/pretrained_models/LargeLanguageModels/llama-3/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa",
    "mistral-7b": "/home/cshdtian/pretrained_models/LargeLanguageModels/mistral-7b/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b",
}


class LanguageModel():
    """
    A basic class for language models.
    """
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generation(self, prompts:List[str], max_seq_len:int, max_batch_size:int):
        """
        Response generation with language models.
        
        Args:
            prompts (List[str]): A list of strings.
            max_seq_len (int): The maximum of the sequence length.
            max_batch_size (int): The maximum of the batch size.
        """
        raise NotImplementedError


class HuggingFace(LanguageModel):
    
    def __init__(self, model_name:str, model, tokenizer):
        """
        Args:
            model_name (str): The name of the adopted language model, e.g., LLama.
            model: The language model.
            tokenizer: The tokenizer.
        """
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_tk_ids = [self.tokenizer.eos_token_id]
    
    def batched_generation(self, 
                           prompts:List[str], 
                           max_seq_len:int, 
                           max_batch_size:int=1,
                           temperature:float=1.0,
                           top_p:float=1.0) -> List[str]:
        """
        Response generation with language models.
        
        Args:
            prompts (List[str]): A list of strings.
            max_seq_len (int): The maximum of the sequence length.
            max_batch_size (int): The maximum of the batch size. Default 1 in our work regarding agent.
            temperature (float): The temperature coefficient of the language model.
        
        Returns:
            A list of responses (the output of the language model).
        """
        if self.model_name == "llama3":
            assert 1 <= max_seq_len <= 8192, f"The maximum of the sequence length is 8192."
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
        
        if temperature > 0.:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_seq_len,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.eos_tk_ids
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_seq_len,
                do_sample=False,
                eos_token_id=self.eos_tk_ids,
                top_p=1,
                temperature=1.0
            )
        
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        
        output_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        for key in inputs:
            inputs[key].to("cpu")
        output_ids.to("cpu")
        del inputs, output_ids
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return output_list
    
    def calculate_perplexity(self, prompts:List[str], max_seq_len:int, max_batch_size:int=1, 
                             temperature:float=1.0, top_p:float=1.0):
        if self.model_name == "llama3":
            assert 1 <= max_seq_len <= 8192, f"The maximum of the sequence length is 8192."
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
        
        if temperature > 0.:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_seq_len,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.eos_tk_ids
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_seq_len,
                do_sample=False,
                eos_token_id=self.eos_tk_ids,
                top_p=1,
                temperature=1.0
            )
        
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        
        outputs = self.model(output_ids, labels=output_ids)
        neg_log_likelihood = outputs.loss
        perplexity = torch.exp(neg_log_likelihood).item()
        
        for key in inputs:
            inputs[key].to("cpu")
        output_ids.to("cpu")
        del inputs, output_ids, outputs
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return perplexity


class LanguageModelSpecification():
    """
    Specify a language model.
    """
    def __init__(self, 
                 model_name:str, 
                 max_seq_len:int, 
                 max_batch_size:int,
                 temperature:float=1.,
                 top_p:float=1.,
                 parallel:bool=False,
                 gpu_id:int=0) -> None:
        
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.top_p = top_p
        self.temperature = temperature
        self.max_seq_len = max_seq_len
        if model_name == "llama-3":
            assert 1 <= self.max_seq_len <= 8192, f"The maximum of the sequence length of LLaMA3 is 8192."
        self.gpu_id = gpu_id
        self.template, self.tokenizer, self.model = self._get_model_and_tokenizer(model_name=model_name, parallel=parallel)
    
    def _get_model_and_tokenizer(self, model_name:str, parallel:bool=False):
        """
        Initialize the language model and tokenizer.
        """
        ### TODO: More language model can be added later.
        if model_name == "llama-3":
            model_id = PATH_DICT["llama-3"]
            template = "llama-3"
        elif model_name == "llama-2":
            model_id = PATH_DICT["llama-2"]
            template = "llama-2"
        elif model_name == "sheared-llama":
            model_id = PATH_DICT["sheared-llama"]
            template = "sheared-llama"
        elif model_name == "mistral-7b":
            model_id = PATH_DICT["mistral-7b"]
            template = "mistral-7b"
        else:
            raise ValueError("Unrecognized model name.")
            
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id,
            use_fast=False
        )
        
        if parallel:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto"
            )
        else:
            device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(device)
            
        model = model.eval()
        
        if model_name == "llama-2":
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "left"
        
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        
        lm = HuggingFace(model_name=self.model_name, 
                         model=model, 
                         tokenizer=tokenizer)
        return template, tokenizer, lm
    
    def response_generation(self, prompt:str):
        conversation = conversation_temp(self.template)
        conversation.append_message(conversation.roles[0], prompt)
        conversation.append_message(conversation.roles[1], None)
        
        full_prompts = conversation.get_prompt()
        #print(full_prompts)

        output = self.model.batched_generation(prompts=full_prompts,
                                               max_seq_len=self.max_seq_len,
                                               temperature=self.temperature,
                                               top_p=self.top_p)

        return output
    
    def calculate_perplexity(self, prompt):
        conversation = conversation_temp(self.template)
        conversation.append_message(conversation.roles[0], prompt)
        conversation.append_message(conversation.roles[1], None)
        
        full_prompts = conversation.get_prompt()
        
        perplexity = self.model.calculate_perplexity(
            prompts=full_prompts,
            max_seq_len=self.max_seq_len,
            temperature=self.temperature,
            top_p=self.top_p
        )
        return perplexity
        
        