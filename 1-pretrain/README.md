# hw1: pretrain

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

[Ссылка на эксперименты на wandb, открывать с vpn](https://wandb.ai/falcon_light/llm-course-pretrain-1?nw=nwuserfalcon_light)

[Ссылка на отчет на wandb (чуть более красивый)](https://api.wandb.ai/links/falcon_light/swam5k7m)

Продублирую некоторые графики

<img width="2528" height="1328" alt="W B Chart 16 11 2025, 14_57_42" src="https://github.com/user-attachments/assets/7643b636-53d0-472b-ba60-f52f086cfab6" />
<img width="2528" height="1328" alt="W B Chart 16 11 2025, 14_58_42" src="https://github.com/user-attachments/assets/a41586b7-904e-4f12-93ae-ecb853bdcd1f" />


Краткие результаты 25 экспериментов:

Наиболее значимые гиперпараметры - lr_scheduler, learning_rate, batch_size

Наименее значимые гипепараметры - gradient_accumulation_stps, torch_compile (ожидаемо, они и не должны сильно влиять на качество)

Лучший ран - 

```python
bf16=false
gradient_accumulation_steps=1
learning_rate=5e-4
learning_rate_scheduler='cosine_with_restarts' (хотя по факту в данном эксперименте это просто cosine)
optim='adamw_apex_fused'
batch_size=16
compile=false
warmup_steps=500
eval_loss = 3.162
```

Примеры генерации:

prompt: "Люди должны читать книги, иначе"

**Baseline**:  Люди должны читать книги, иначе 1999 сын решался связанных горя� эпил сохраня Union damage напряженно причал books беседымак themselves первич ответил всплы риф Сибири задер путе артефак наличиевека Дальнего большего семжан ярмар alongither получено Bo ман критике ящик КА Иванов требует Маг записа земной

**Trained model**:  Люди должны читать книги, иначе это обстоятельство, в то время как его коллеги, и все.

Упоминается в «Огон» и был признан «Ходором» художником-постановщиком. В его честь есть два сына


## Выводы и наблюдения
- Высокий learning_rate и batch_size уменьшают grad_norm
- Высокий grad_norm -> высокий train_loss -> высокий eval_loss
- В целом, train_loss и eval_loss очень сильно скореллированы
- В данном сетапе, низкий learning_rate (< 1e-4) вредит, ничего не успевает сходиться
- Чтобы получить более значимые инсайты по другим параметрам, возможно стоит провести больше экспериментов (так как пространство признаков очень большое)
