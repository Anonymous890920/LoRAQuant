from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import PeftModel
import torch
import argparse 
import os
from peft import get_peft_model_state_dict
from tqdm import tqdm
from quantization_utils import quantize
from pbllm import pbllm
from billm import billm
from gptq import gptq_quant
import random
import json
from datasets import load_dataset
import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="train models with lora")

    parser.add_argument(
        "--model_name",
        choices=["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-13b-hf"],
        required=True
    )
    parser.add_argument(
        "--adapter_path",
        required=True,
        type=str
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default='gsm8k',
        choices=["gsm8k", "minerva_math_algebra", "humaneval", "mbpp", "EdinburghNLP/xsum"]
    )
    
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0
    )

    parser.add_argument(
        "--high_rank",
        type=int
    )

    parser.add_argument(
        "--num_bits_high",
        type=int
    )

    parser.add_argument(
        "--num_bits_low",
        type=int
    )

    parser.add_argument(
        "--group_size",
        type=int,
        default=128
    )
    
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.7
    )
    
    parser.add_argument(
        "--along_column_B",
        action='store_true'
    )
    
    parser.add_argument(
        "--along_column_A",
        action='store_true'
    )
        
    parser.add_argument(
        "--method",
        choices=['fp', 'rtn', 'bin', 'pbllm', 'gptq', 'loraquant_svd', 'loraquant_ratio', 'loraquant_random', 'loraquant_norm', 'billm'],
        required=True
    )
    
    parser.add_argument(
        "--opt",
        action='store_true'
    )

    parser.add_argument(
        "--pbllm_low_frac",
        type=float,
        default=0.1
    )
    
    args = parser.parse_args()
    return args

def optimize(B, A , num_bits=2, group_size=128, along_column_B=False, along_column_A=False, steps=100, lr=1e-2, method='rtn'):
    B_ = torch.nn.Parameter(B.clone()) 
    A_ = torch.nn.Parameter(A.clone())

  
    l = torch.norm(B@A - quantize(B_, group_size=group_size, num_bits=num_bits, along_column=along_column_B, method=method) \
                              @ quantize(A_, group_size=group_size, num_bits=num_bits, along_column=along_column_A, method=method), p='fro') 
 
    optimizer = torch.optim.SGD([B_, A_], lr=lr)
    # pbar = tqdm(range(steps), desc="Optimizing", dynamic_ncols=True)
    for _ in range(steps):
      
        loss = torch.norm(B@A - quantize(B_, group_size=group_size, num_bits=num_bits, along_column=along_column_B, method=method) \
                              @ quantize(A_, group_size=group_size, num_bits=num_bits, along_column=along_column_A, method=method), p='fro') 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        # pbar.set_description(f"Loss: {loss.item():.6f}")
 
    original_error = torch.norm(B@A - quantize(B, group_size=group_size, num_bits=num_bits, along_column=along_column_B, method=method) \
                              @ quantize(A, group_size=group_size, num_bits=num_bits, along_column=along_column_A, method=method), p='fro') 
    new_error = torch.norm(B@A - quantize(B_, group_size=group_size, num_bits=num_bits, along_column=along_column_B, method=method) \
                              @ quantize(A_, group_size=group_size, num_bits=num_bits, along_column=along_column_A, method=method), p='fro') 
    

    print(f"original error = {original_error}")
    print(f"new_error \t= {new_error}")
    return B_.detach(),A_.detach()

def loraquant(lora_B, lora_A, high_rank=4, num_bits_high=2, 
                  group_size=128,  along_column_B=True, along_column_A=False, method='loraquant_svd', ratio=0.7, opt=False):
    
    m = lora_B.shape[0]
    r = lora_B.shape[1]
    n = lora_A.shape[1]
    
    if method in ['loraquant_svd', 'loraquant_ratio']:
        U, S, Vh = torch.linalg.svd((lora_B@lora_A))
        U = U[:, 0:r]
        S = S[0:r]
        Vh = Vh[0:r, :]
        
        if method == 'loraquant_ratio':
            S_squared = S**2
            total_variance = S_squared.sum()
            cumulative_variance = torch.cumsum(S_squared, dim=0)
            explained = cumulative_variance / total_variance
            high_rank = (explained >= ratio).nonzero(as_tuple=True)[0][0].item()
            
            if high_rank == 0:
                high_rank = 1
            
        B_high = (U[:, 0:high_rank] @ torch.diag(torch.sqrt(S[0:high_rank])))
        A_high = (torch.diag(torch.sqrt(S[0:high_rank])) @ Vh[0:high_rank, :])
                
        B_low = (U[:, high_rank:r] @ torch.diag(torch.sqrt(S[high_rank:r])))
        A_low = (torch.diag(torch.sqrt(S[high_rank:r])) @ Vh[high_rank:r, :])
    elif method == 'loraquant_random':
        perm = list(range(r))
        random.shuffle(perm)
        B_high = lora_B[:, perm[0:high_rank]]
        A_high = lora_A[perm[0:high_rank], :]
        B_low = lora_B[:, perm[high_rank:]]
        A_low = lora_A[perm[high_rank:], :]
    elif method == 'loraquant_norm':
        lst = [torch.norm(lora_B[:,i].reshape((-1,1))@lora_A[i,:].reshape((1,-1))) for i in range(r)]
        perm = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
        B_high = lora_B[:, perm[0:high_rank]]
        A_high = lora_A[perm[0:high_rank], :]
        B_low = lora_B[:, perm[high_rank:]]
        A_low = lora_A[perm[high_rank:], :]
    
    if opt:
        for i in range(high_rank):
                B_, A_ = optimize(B_high[:, i:i+1], A_high[i:i+1, :] , num_bits=num_bits_high, group_size=group_size, along_column_B=along_column_B, along_column_A=along_column_A, steps=300, lr=1e-2)
                B_high[:, i:i+1] = B_
                A_high[i:i+1, :] = A_
    # elif method == 'loraquant_opt_test':
        # B_,A_ = optimize(B_high, A_high, num_bits=num_bits_high, group_size=group_size, along_column_B=along_column_B, along_column_A=along_column_A, steps=300, lr=1e-2)
        # B_high = B_
        # A_high = A_
                
    B_low_quantized = quantize(B_low, group_size=group_size, num_bits=1, along_column=along_column_B, method='bin')
    A_low_quantized = quantize(A_low, group_size=group_size, num_bits=1, along_column=along_column_A, method='bin')
    B_high_quantized = quantize(B_high, group_size=group_size, num_bits=num_bits_high, along_column=along_column_B, method='rtn')
    A_high_quantized = quantize(A_high, group_size=group_size, num_bits=num_bits_high, along_column=along_column_A, method='rtn')
    
    B = torch.cat((B_high_quantized, B_low_quantized), dim=1)
    A = torch.cat((A_high_quantized, A_low_quantized), dim=0)
    
    
    return B, A, high_rank * (m+n) * num_bits_high + (r-high_rank)*(m+n), r*(m+n)

def get_output_name(args):
    if args.method == 'fp':
        output_path = f'result/{args.model_name}/{args.adapter_path}/{args.dataset}/fp_numfewshot{args.num_fewshot}.json'
    elif args.method in ['rtn', 'bin']:
        output_path = f'result/{args.model_name}/{args.adapter_path}/{args.dataset}/{args.method}_{args.num_bits_low}bit_numfewshot{args.num_fewshot}.json'
    elif args.method == 'pbllm':
        output_path = f'result/{args.model_name}/{args.adapter_path}/{args.dataset}/pbllm_{args.pbllm_low_frac}_{args.num_bits_high}bit_numfewshot{args.num_fewshot}.json'
    elif args.method == 'billm':
        output_path = f'result/{args.model_name}/{args.adapter_path}/{args.dataset}/billm.json'
    elif args.method == 'gptq':
        output_path = f'result/{args.model_name}/{args.adapter_path}/{args.dataset}/gptq_{args.num_bits_high}bit_numfewshot{args.num_fewshot}.json'
    elif args.method == 'loraquant_svd':
        output_path = f'result/{args.model_name}/{args.adapter_path}/{args.dataset}/loraquant_svd_h{args.high_rank}_{args.num_bits_high}_opt{args.opt}_Bcol{args.along_column_B}_Acol{args.along_column_A}_numfewshot{args.num_fewshot}.json'
    elif args.method == 'loraquant_ratio':
        output_path = f'result/{args.model_name}/{args.adapter_path}/{args.dataset}/loraquant_ratio_r{args.ratio}_{args.num_bits_high}_opt{args.opt}_Bcol{args.along_column_B}_Acol{args.along_column_A}_numfewshot{args.num_fewshot}.json'
    elif args.method == 'loraquant_random':
        output_path = f'result/{args.model_name}/{args.adapter_path}/{args.dataset}/loraquant_random_h{args.high_rank}_{args.num_bits_high}_opt{args.opt}_Bcol{args.along_column_B}_Acol{args.along_column_A}_numfewshot{args.num_fewshot}.json'
    elif args.method == 'loraquant_norm':
        output_path = f'result/{args.model_name}/{args.adapter_path}/{args.dataset}/loraquant_norm_h{args.high_rank}_{args.num_bits_high}_opt{args.opt}_Bcol{args.along_column_B}_Acol{args.along_column_A}_numfewshot{args.num_fewshot}.json'
    elif args.method == 'jddiag':
        output_path = f'result/{args.model_name}/{args.adapter_path}/{args.dataset}/jddiag.json'
    else:
        output_path = f'result/{args.model_name}/{args.adapter_path}/{args.dataset}/{args.method}_lowfrac{args.pbllm_low_frac}_h{args.high_rank}_{args.high_quant}_numfewshot{args.num_fewshot}.json'
    return output_path
    
def main():
    args = parse_args()
    random.seed(42)
    ## load model ##
    access_token = "replace with huggingface token to access gated models"
    
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
   
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name, quantization_config=bnb_config, token=access_token
    ).to('cuda')
    print('model loaded')
 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token)

    model = PeftModel.from_pretrained(model, args.adapter_path, local_files_only=True)
    model.eval()
    
    
    lora_state_dict = get_peft_model_state_dict(model)
    
    if args.method == 'fp':
        pass
    elif args.method in ['rtn', 'bin']:
        for lora_A in lora_state_dict:
            if 'lora_B' in lora_A:
                continue
            lora_A_name = lora_A.replace('lora_A.weight', 'lora_A.default')
            lora_B_name = lora_A.replace('lora_A.weight', 'lora_B.default')
            print(f'quantizing {lora_B_name}, {lora_A_name}')
            lora_B = model.get_submodule(lora_B_name).weight
            lora_A = model.get_submodule(lora_A_name).weight
            
            lora_B.data = quantize(lora_B, group_size=args.group_size, num_bits=args.num_bits_low, method=args.method)
            lora_A.data = quantize(lora_A, group_size=args.group_size, num_bits=args.num_bits_low, method=args.method)
    elif args.method == 'pbllm':
        pbllm(model, tokenizer, group_size=args.group_size, num_bits_high=args.num_bits_high, low_frac=args.pbllm_low_frac, nsamples=128)
    elif args.method == 'billm':
        billm(model, tokenizer, group_size=args.group_size, nsamples=128)
    elif args.method == 'gptq':
        gptq_quant(model, tokenizer, group_size=args.group_size, num_bits=args.num_bits_high,  nsamples=128)
    elif 'loraquant' in args.method:
        total_bits = 0
        total_params = 0
        for lora_A in tqdm(lora_state_dict):
            if 'lora_B' in lora_A:
                continue
            lora_A_name = lora_A.replace('lora_A.weight', 'lora_A.default')
            lora_B_name = lora_A.replace('lora_A.weight', 'lora_B.default')
            print(f'quantizing {lora_B_name}, {lora_A_name}')
            
            lora_B = model.get_submodule(lora_B_name).weight
            lora_A = model.get_submodule(lora_A_name).weight
            
            B, A, num_bits, num_params = loraquant(lora_B, lora_A, method=args.method, high_rank=args.high_rank, num_bits_high=args.num_bits_high, group_size=args.group_size,  along_column_B=args.along_column_B, along_column_A=args.along_column_A, ratio=args.ratio, opt=args.opt)
            
            total_bits += num_bits
            total_params += num_params
            lora_B.data = B.clone()
            lora_A.data = A.clone()
        print(f'avgbit = {total_bits/total_params}, {args.method}, {args.ratio}, {args.model_name}')

    if args.dataset in ['humaneval', 'mbpp']:
        import tempfile

        from accelerate import Accelerator
        from accelerate.utils import write_basic_config

        from bigcode_eval.arguments import EvalArguments
        from bigcode_eval.evaluator import Evaluator

        def update_args(args_bigcode):
            # the executed code for the tests is safe (see tests/data/*_eval_gens.json)
            args_bigcode.allow_code_execution = True
            args_bigcode.save_generations = False
            args_bigcode.save_generations_path = ""
            args_bigcode.save_references = False
            args_bigcode.save_references_path = ""
            args_bigcode.metric_output_path = TMPDIR
            args_bigcode.load_generations_path = None
            args_bigcode.generation_only = False
            args_bigcode.check_references = False
            # postprocessing for HumanEval and MBPP makes generations
            # with dummy model not distinctive
            args_bigcode.postprocess = True
            args_bigcode.instruction_tokens = None
            args_bigcode.limit = 1
            args_bigcode.limit_start = 0
            args_bigcode.batch_size = 1
            args_bigcode.max_length_generation = 1024 if args.dataset=='humaneval' else 2048
            args_bigcode.left_padding = False
            args_bigcode.do_sample = False
            args_bigcode.top_p = 0
            args_bigcode.n_samples = 1
            args_bigcode.seed = 0
            args_bigcode.prompt = None
            args_bigcode.precision = None
            args_bigcode.modeltype = None
            args_bigcode.max_memory_per_gpu = None
            # args_bigcode.temperature = 0.2
            # args_bigcode.top_p = 0.95
            args_bigcode.prefix = "Make this code work in Python:" 
            # args_bigcode.prompt = 'def'
            return args_bigcode

        gens = []
        TMPDIR = f'bigcode_{args.dataset}'
        args_bigcode = update_args(EvalArguments())
        set_seed(args_bigcode.seed)
        tokenizer.pad_token = tokenizer.eos_token
        
        for limit_start in range(164):
            args_bigcode.limit_start = limit_start


            configPath = os.path.join(TMPDIR, "default_config.yml")
            write_basic_config(save_location=configPath)
            accelerator = Accelerator()

            args_bigcode.generation_only = True
            args_bigcode.save_every_k_tasks = -1
            evaluator = Evaluator(accelerator, model, tokenizer, args_bigcode)
            generations, references = evaluator.generate_text('humaneval')
            gens.append([generations[0][0]])
        output_path = get_output_name(args)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(gens, f)
            
        args_bigcode.limit = None
        args_bigcode.limit_start = 0
        args_bigcode.load_generations_path = output_path
        evaluator = Evaluator(accelerator, model, tokenizer, args_bigcode)
        results = evaluator.evaluate('humaneval')
        print(results)
    
        with open(output_path, "w") as f:
            json.dump({'results':results, 'gens':gens}, f, indent=0) 
    elif args.dataset == 'EdinburghNLP/xsum':
        ds = load_dataset("EdinburghNLP/xsum")['test']
        gens = []
        refs = []

        for data in tqdm(ds):

            doc = data['document']
            ref = data['summary']
            
            prompt = f"Document : {doc} \n\n Summary : "


            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            refs.append(ref)
            gens.append(tokenizer.decode(new_tokens, skip_special_tokens=True))



        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=gens, references=refs)
        print(results)
        output_path = get_output_name(args)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({'results':results, 'gens':gens, 'refs':refs}, f, indent=0) 
    else:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM


        
        hf_lm = HFLM(pretrained=model, tokenizer=tokenizer, device="cuda")

            # Run the evaluation
        results = evaluator.simple_evaluate(
                model=hf_lm,
                tasks=[args.dataset],
                num_fewshot=args.num_fewshot,
                batch_size=1,
                device="cuda"
        )

    
        output_path = get_output_name(args)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({'results':results['results'], 'gens':results['samples'][args.dataset]}, f, indent=0) 
    
if __name__ == "__main__":
    main()