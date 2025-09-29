from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel,PeftConfig, get_peft_model_state_dict
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import evaluate
from tqdm import tqdm

from .binary import Binarization
from .bigptq import BRAGPTQ

from .datautils import get_loaders

def find_layers(module, layers=[torch.nn.Conv2d, torch.nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

'''
The function is employed to calibrate and quantize models layer by layer.
'''
@torch.no_grad()
def quant_sequential(model, dataloader, nsamples, seqlen, group_size=128, dev='cuda', low_quant_method='braq'):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "llama" in model.name_or_path:
        layers = model.model.model.layers
    elif "mistral" in model.name_or_path:
        layers = model.model.model.layers
        
    print("Ready.")
    
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
    
        gptq = {}
        for name in subset:
            if 'lora_A.default' in name or 'lora_B.default' in name:
                braq_quantizer = Binarization(
                    subset[name].weight,
                    method=low_quant_method,
                    groupsize=group_size,
                )
                gptq[name] = BRAGPTQ(
                    subset[name],
                    braq_quantizer,
                    salient_metric='hessian',
                    disable_gptq=False,
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        for batch in dataloader:
           model(batch[0].to(dev))
            
   
        for h in handles:
            h.remove()

        for name in gptq:
            print(i, name)
            print("Quantizing ...")
            info = gptq[name].fasterquant(
                percdamp=0.01, 
                blocksize=group_size,
            )
            gptq[name].free()
            
        del layer
        del gptq

    model.config.use_cache = use_cache
    



def billm(model, tokenizer, method='braq', group_size=128, nsamples=128, low_quant_method='braq'):
    assert method in ['braq', 'gptq']
    
    if 'llama' in model.name_or_path:
        model.seqlen = 2048
    elif 'mistra' in model.name_or_path:
        model.seqlen = 2048
        
    dataloader, testloader = get_loaders(
            'wikitext2',
            tokenizer=tokenizer,
            nsamples=nsamples, # number of calibration data
            seed=1234,
            seqlen=model.seqlen,
    )
    
    quant_sequential(model, dataloader, nsamples=nsamples, seqlen=model.seqlen, low_quant_method=low_quant_method, group_size=group_size)
    

    
