from .gptq import *
from .datautils import *
from .quant import *
import torch

def find_layers(module, layers=[torch.nn.Conv2d, torch.nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def quant_sequential(model, dataloader, nsamples=128, num_bits=2, group_size=128):
    model.eval()
    
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    if "llama" in model.name_or_path:
        layers = model.model.model.layers
    elif "mistral" in model.name_or_path:
        layers = model.model.model.layers
    elif "gemma" in model.name_or_path:
        layers = model.model.model.layers

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i]
        full = find_layers(layer)
        
        for name in full:
            if 'lora_A.default' in name or 'lora_B.default' in name:
                subset = {name: full[name]}

                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer = Quantizer()
                    gptq[name].quantizer.configure(
                        num_bits, perchannel=True, sym=True, mse=False
                    )

                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq[name].add_batch(inp[0].data, out.data)
                    return tmp
                handles = []
                
             
        
        
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for batch in dataloader:
                    model(batch[0].to('cuda'))
                for h in handles:
                    h.remove()

                for name in subset:
                    print('Quantizing ...',i, name)

                    gptq[name].fasterquant(
                        percdamp=0.1, groupsize=group_size, actorder=True, static_groups=False
                    )
                    quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                    gptq[name].free()

  
        layers[i] = layer.cuda()
        del layer
        del gptq 
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    
    return quantizers


def gptq_quant(model, tokenizer, group_size=128, num_bits=8,  nsamples=128):
    if 'llama' in model.name_or_path:
        model.seqlen = 2048
    elif 'mistral' in model.name_or_path:
        model.seqlen = 2048
    elif 'gemma' in model.name_or_path:
        model.seqlen = 2048
        
    dataloader, _ = get_loaders(
            'wikitext2',
            tokenizer=tokenizer,
            nsamples=nsamples, # number of calibration data
            seed=1234,
            seqlen=model.seqlen,
    )
    
    quant_sequential(model, dataloader, nsamples=128, num_bits=num_bits, group_size=group_size)