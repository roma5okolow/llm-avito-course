from transformers import AutoModelForCausalLM
import torch
from solution import create_model, prepare_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = prepare_tokenizer()
model_baseline = create_model(tokenizer).to(device)
model_trained = AutoModelForCausalLM.from_pretrained('output_dir/gpt2-1b-russian/run-vvksou9y/checkpoint-5000').to(device)


def generate_text(model, prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.9):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

prompt = "Люди должны читать книги, иначе"
print('Baseline: ', generate_text(model_baseline, prompt))
print('-' * 50)
print('Trained model: ', generate_text(model_trained, prompt))