# LoRAQuant

LoRAQuant provides efficient quantization techniques for LoRA adapters.  

### Download Weights
Pre-trained LoRA weights used in our experiments are available [here](https://mega.nz/folder/0ZpSxTBB#nYyDkqxouidzHmAZYhGPPA).  

### Evaluation
You can evaluate a model with our method using the following command:

```bash
python main.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --adapter_path path_to_adapter_weights \
    --dataset gsm8k \
    --group_size 128 \
    --method lorauquant_ratio \
    --num_bits_low 1 \
    --num_bits_high 2 \
    --ratio 0.8 \
    --num_fewshot 0 \
    --along_column_B \
    --opt
