from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer,DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import argparse 
import os

def parse_args():
    parser = argparse.ArgumentParser(description="train models with lora")

    parser.add_argument(
        "--model",
        choices=["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-13b-hf"],
        required=True
    )
    parser.add_argument(
        "--dataset",
        choices=["meta-math/MetaMathQA", "ise-uiuc/Magicoder-Evol-Instruct-110K", "WizardLMTeam/WizardLM_evol_instruct_70k", "EdinburghNLP/xsum"],
        required=True   
    )
    parser.add_argument(
        "--lora",
        action='store_true'
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4
    )
    args = parser.parse_args()
    return args


def get_train_val_dataset(dataset_name, max_length, tokenizer):
    def preprocess_magicoder(example):
        inputs = [instruction + response
                for instruction,response in zip(example['instruction'], example['response'])]
        return tokenizer(inputs, truncation=True, max_length=max_length, padding="max_length")
    
    def preprocess_humaneval(example):
        inputs = [instruction + response
                for instruction,response in zip(example['prompt'], example['canonical_solution'])]
        return tokenizer(inputs, truncation=True, max_length=max_length, padding="max_length")
    
    def preprocess_metamath(example):
        inputs = [f'Question: {question}\n Answer: {answer}\n\n'
                for question,answer in zip(example['query'], example['response'])]
        return tokenizer(inputs, truncation=True, max_length=max_length, padding="max_length")
    
    def preprocess_gsm8k(example):
        inputs = [f'Question: {question}\n Answer: {answer}\n\n'
                for question,answer in zip(example['question'], example['answer'])]
        return tokenizer(inputs, truncation=True, max_length=max_length, padding="max_length")
    
    def preprocess_xsum(example):
        inputs = [f'Document : {document} \n\n Summary : {summary}'
                for document,summary in zip(example['document'], example['summary'])]
        return tokenizer(inputs, truncation=True, max_length=max_length, padding="max_length")


    train_data = load_dataset(dataset_name)['train']

    if dataset_name =='meta-math/MetaMathQA':
        val_data = load_dataset("openai/gsm8k", "main")['test']
        print(train_data.column_names)
        train_data = train_data.map(preprocess_metamath, batched=True, remove_columns=train_data.column_names)
        val_data = val_data.map(preprocess_gsm8k, batched=True, remove_columns=val_data.column_names)
        
    elif dataset_name=='ise-uiuc/Magicoder-Evol-Instruct-110K':
        val_data = load_dataset("openai/openai_humaneval")['test']
        print(train_data.column_names)
        train_data = train_data.map(preprocess_magicoder, batched=True, remove_columns=train_data.column_names)
        val_data = val_data.map(preprocess_humaneval, batched=True, remove_columns=val_data.column_names)
        
    elif dataset_name=="WizardLMTeam/WizardLM_evol_instruct_70k":
        val_data = load_dataset("openai/openai_humaneval")['test']
 
        train_data = train_data.map(preprocess_wizardlm, batched=True, remove_columns=train_data.column_names)
        val_data = val_data.map(preprocess_humaneval, batched=True, remove_columns=val_data.column_names)
    elif dataset_name=="EdinburghNLP/xsum":
        val_data = load_dataset("EdinburghNLP/xsum")['test']
        train_data = train_data.map(preprocess_xsum, batched=True, remove_columns=train_data.column_names)
        val_data = val_data.map(preprocess_xsum, batched=True, remove_columns=val_data.column_names)
        
    print(dataset_name)
    return train_data, val_data

def main():
    args = parse_args()

    # os.environ["WANDB_MODE"] = "offline"
    
    ### load model ###
    access_token = "replace with huggingface token to access gated models"
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model,quantization_config=bnb_config, device_map={"": 0},  token=access_token)
    print(model)
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, add_eos_token=True,  token=access_token)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.lora:
        peft_config = LoraConfig(
                    lora_alpha=2 * args.rank,
                    lora_dropout=0,
                    r=args.rank,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules='all-linear'
        )
        model = get_peft_model(model, peft_config)

    ### load dataset ### 
    train_data, val_data = get_train_val_dataset(args.dataset, args.max_length, tokenizer)

    training_args = TrainingArguments(
            output_dir=args.output_dir,
            eval_strategy="steps",
            optim="adamw_torch",
            save_strategy="epoch",
            log_level="debug",
            logging_steps=10,
            eval_steps=7000,
            fp16=True,
            do_eval=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            weight_decay=0,
            num_train_epochs=args.num_train_epochs, 
            learning_rate=args.lr,
            report_to="tensorboard"
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    # Save the model
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
