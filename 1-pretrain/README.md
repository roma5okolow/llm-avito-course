# hw1

Подбирал гиперпараметры для претрейна через wandb sweep.
Мой конфиг:
``` yml
{
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
```
Эксперименты проводились на A100 80 GB.

Примеры генерации:

prompt: "Люди должны читать книги, иначе"

**Baseline**:  Люди должны читать книги, иначе 1999 сын решался связанных горя� эпил сохраня Union damage напряженно причал books беседымак themselves первич ответил всплы риф Сибири задер путе артефак наличиевека Дальнего большего семжан ярмар alongither получено Bo ман критике ящик КА Иванов требует Маг записа земной

**Trained model**:  Люди должны читать книги, иначе это обстоятельство, в то время как его коллеги, и все.

Упоминается в «Огон» и был признан «Ходором» художником-постановщиком. В его честь есть два сына

 