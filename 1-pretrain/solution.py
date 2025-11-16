import os
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import torch
import wandb
import time
from pathlib import Path

# отказался от WandbSettings, чтобы заработал sweep
os.environ['HTTP_PROXY'] = os.getenv('AVITO_HTTP_PROXY', '')
os.environ['HTTPS_PROXY'] = os.getenv('AVITO_HTTPS_PROXY', '')

# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 30
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer) -> dict:
    output = tokenizer(
        text=examples['text'],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    # -100 взял из https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
    # таргет не сдвигаем, так как это должна делать моедль внутри себя
    labels = torch.where(output[ATTENTION_MASK] == 1, output[INPUT_IDS], -100)
    return {
        LABELS: labels,
        INPUT_IDS: output[INPUT_IDS],
        ATTENTION_MASK: output[ATTENTION_MASK]
    }



def save_as_parquets(ds: Dataset, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    os.makedirs(output_dir, exist_ok=True)
    for index in range(num_shards):
        ds.shard(num_shards, index).to_parquet(Path(output_dir) / f'{index:05d}.parquet')

def prepare_dataset():
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, num_proc=32)
    save_as_parquets(dataset)



def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    data_files = [
        str(Path(data_dir) / file) for file in os.listdir(data_dir)
        if file.endswith('parquet')
    ]
    return load_dataset('parquet', data_files=data_files, split='train')


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(tokenizer):
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )
    
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model


def initialize_wandb():
    wandb.init(
        project="llm-course-pretrain-1",
        name="bs",
    )

def wandb_hp_space(trial):
    return {
        "method": "bayes",
        "project": "llm-course-pretrain-1",
        "metric": {"name": "final_eval_loss", "goal": "minimize"},
        "parameters": {
            'learning_rate': {'min': 1e-5, 'max': 1e-3, 'distribution': 'log_uniform_values'},
            "warmup_steps": {"values": [50, 200, 500, 1000]},
            "lr_scheduler_type": {
                "values": ["linear", "constant_with_warmup", "cosine", 'cosine_with_restarts']
            },
            "per_device_train_batch_size": {
                "values": [4, 8, 16]
            },
            "gradient_accumulation_steps": {
                "values": [1, 2, 4]
            },
            "torch_compile": {
                "values": [True, False]
            },
            "optim": {
                "values": ["adamw_torch", "adamw_apex_fused", "adafactor"]
            },
            "bf16": {
                "values": [True, False]
            },
        },
    }

TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch',
    'num_train_epochs': 0.1,
    'save_steps': 1000,
    'save_total_limit': 20,
    'weight_decay': 0.01,
    'logging_steps': 10,
    'eval_steps': 1000,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'gradient_checkpointing': False,
    'dataloader_num_workers': 32,
    'torch_compile': True,
    'report_to': 'wandb',
}


def train_model():
    tokenizer = prepare_tokenizer()
    train_dataset, eval_dataset = split_dataset(load_tokenized_dataset())
    training_args = TrainingArguments(**TRAINING_CONFIG)
    trainer = Trainer(
        model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_init=lambda x: create_model(tokenizer),
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)] # dont change
        )
    
    trainer.hyperparameter_search(
        direction="minimize",
        backend="wandb",
        hp_space=wandb_hp_space,
        n_trials=25,
    ) 

if __name__ == "__main__":
    # Step 1: Prepare the dataset (run once)
    # prepare_dataset()
    
    # Step 2: Train the model
    train_model()