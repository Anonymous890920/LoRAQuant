from .gptq import LowHighGPT
from .high_quant import HighQuantizer
from .low_quant import LowQuantizer
from .datautils import get_loaders
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
def quant_sequential(model, dataloader, group_size=128, high_bit=8, salient_metric='hessian', low_frac=0.5):
    model.eval()
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "llama" in model.name_or_path:
        layers = model.model.model.layers
    elif "mistral" in model.name_or_path:
        layers = model.model.model.layers

    print("Ready.")
  
    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            if 'lora_A.default' in name or 'lora_B.default' in name:
                low_quantizer = LowQuantizer(
                    subset[name].weight,
                    method='xnor',
                    groupsize=group_size,
                )
                high_quantizer = HighQuantizer(
                    high_bit,
                    True,
                    False,
                    False,
                )
                gpts[name] = LowHighGPT(
                    subset[name],
                    low_quantizer,
                    high_quantizer,
                    salient_metric=salient_metric,
                    disable_gptq=True,
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for batch in dataloader:
           model(batch[0].to('cuda'))
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Quantizing ...")
            info = gpts[name].fasterquant(
                low_frac, percdamp=0.01, blocksize=group_size # todo check this
            )
            gpts[name].free()
    


        layers[i] = layer.to('cuda')
        del layer
        del gpts
        torch.cuda.empty_cache()

      
    model.config.use_cache = use_cache
    
    
def pbllm(model, tokenizer, group_size=128, num_bits_high=8, low_frac=0.5, nsamples=128):
    
    if 'llama' in model.name_or_path:
        model.seqlen = 2048
    elif 'mistral' in model.name_or_path:
        model.seqlen = 2048 # todo check this
        
    dataloader, testloader = get_loaders(
            'wikitext2',
            tokenizer=tokenizer,
            nsamples=nsamples, # number of calibration data
            seed=1234,
            seqlen=model.seqlen,
    )
    
    quant_sequential(model, dataloader, group_size=group_size, low_frac=low_frac, high_bit=num_bits_high)
