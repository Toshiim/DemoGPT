import math
import torch.nn as nn
import inspect
import torch

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # Flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dim

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=True)
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfig:
    """ Base GPT config, params common to all GPT versions """


    def __init__(self, vocab_size, block_size, n_layer=6, n_head=6, n_embd=384, dropout=0.1, bias=False):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("Number of parameters:", self.get_num_params() / 1e6, "M")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :])  # Use this if you want to speed up inference, otherwise use the line below
            # logits = self.lm_head(x)
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # This is a simplified version, you'll likely load your own checkpoint
        # ... (implementation details for loading from checkpoint)
        pass

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Начинаем с итерации по всем именованным параметрам модели
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Фильтруем те параметры, для которых требуется grad (не замороженные)
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Создаем группы параметров: те, что будут подвергаться weight decay, и те, что не будут
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Создаем оптимизатор AdamW
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # First estimate the number of flops we do per iteration.
        # See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_achieved = flops_per_fwdbwd * fwdbwd_per_iter / dt  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


from tokenizers import Tokenizer

tokenizer_path = 'data/out_txt/tokenizer.json'
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()
print(f"Loaded tokenizer with vocab size: {vocab_size}")


config = GPTConfig(vocab_size=vocab_size, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1, bias=False)

# 3. Создание модели
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT(config).to(device)
print(f"Model initialized on device: {device}")

# 4. Загрузка данных (пример с torch Dataset и DataLoader)
import numpy as np
from torch.utils.data import Dataset, DataLoader


# В train.py, в классе TokenizedDataset

class TokenizedDataset(Dataset):
    def __init__(self, data_dir, block_size):
        """
        Инициализирует датасет.
        - data_dir: Папка, где лежат train_fragments.npy и meta.json.
        - block_size: Размер контекста (должен совпадать с max_length из скрипта токенизации).
        """
        self.block_size = block_size

        # 1. Загружаем метаданные, чтобы узнать форму массива
        meta_path = os.path.join(data_dir, 'meta.json')
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл meta.json не найден в папке {data_dir}. Он необходим для загрузки memmap.")

        # Проверяем, совпадает ли block_size с тем, что был при создании файла
        if meta['max_length'] != self.block_size:
            raise ValueError(
                f"Ошибка! block_size ({self.block_size}) не совпадает с max_length в meta.json ({meta['max_length']})")

        # 2. Определяем параметры для memmap
        data_path = os.path.join(data_dir, 'train_fragments.npy')
        num_fragments = meta['num_fragments']
        # Важно: dtype должен быть тот же, что и при сохранении! В нашем скрипте это np.int64
        dtype = np.int64

        # 3. Открываем файл с помощью memmap в режиме только для чтения
        # Файл не будет загружен в RAM. ОС будет сама подгружать нужные страницы с диска.
        try:
            self.data = np.memmap(data_path, dtype=dtype, mode='r', shape=(num_fragments, self.block_size))
            print(f"Файл {data_path} успешно открыт с помощью memmap.")
            print(f"Dataset shape: {self.data.shape}, dtype: {self.data.dtype}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл данных {data_path} не найден.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Получаем срез из memmap-файла. На этом этапе происходит чтение с диска.
        # Это очень быстро, так как ОС эффективно кеширует дисковые операции.
        fragment = self.data[idx]

        # Конвертируем в int64 для PyTorch (требование для embedding слоев)
        full_seq = torch.from_numpy(fragment.astype(np.int64)) #  В новых версиях токенизатор сохраняеться в int64, поэтому надо убрать astype

        # Разделяем на вход (x) и цель (y)
        x = full_seq[:-1]
        y = full_seq[1:]

        return x, y

import torch.nn.functional as F
from tokenizers import Tokenizer

def worker_init_fn(worker_id):
    torch.manual_seed(42 + worker_id)


import os
import time
import json
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer


def train_model_with_saving():
    """Улучшенный цикл обучения с сохранением и мониторингом"""

    scaler = torch.amp.GradScaler('cuda')

    # Загрузка модели и данных
    tokenizer = Tokenizer.from_file('data/out_txt/tokenizer.json')
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(vocab_size=vocab_size, block_size=256, n_layer=6,
                       n_head=6, n_embd=384, dropout=0.1, bias=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(config).to(device)

    # Датасет и даталоадер
    dataset = TokenizedDataset('data/out_txt/', config.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=3,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )

    # Оптимизатор и планировщик
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4,
                                           betas=(0.9, 0.95), device_type=device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Создаем папки
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Файл для логов
    log_file = f'logs/training_log_{int(time.time())}.jsonl'

    # Переменные для отслеживания
    best_loss = float('inf')
    patience = 0
    max_patience = 10

    # Статистика по обучению
    training_start_time = time.time()
    global_step = 0

    print("Начинаем обучение...")
    print(f"Логи сохраняются в: {log_file}")

    for epoch in range(100):  # Максимум 100 эпох
        model.train()
        epoch_start_time = time.time()
        epoch_losses = []

        for i, (x, y) in enumerate(dataloader):
            batch_start_time = time.time()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward pass с AMP
            with torch.amp.autocast('cuda'):
                logits, loss = model(x, y)

            # Backward pass с AMP
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_time = time.time() - batch_start_time
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            global_step += 1

            # Логирование каждые 100 батчей
            if i % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]

                # Быстрое логирование в файл
                log_entry = {
                    "type": "batch",
                    "epoch": epoch,
                    "batch": i,
                    "global_step": global_step,
                    "loss": loss_value,
                    "lr": current_lr,
                    "batch_time": batch_time,
                    "timestamp": time.time()
                }

                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

                print(
                    f"Epoch {epoch}, Batch {i}, Loss: {loss_value:.4f}, LR: {current_lr:.6f}, Time: {batch_time:.3f}s")

        # Статистика по эпохе
        epoch_time = time.time() - epoch_start_time
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        scheduler.step()

        # Логирование эпохи
        epoch_log = {
            "type": "epoch",
            "epoch": epoch,
            "avg_loss": avg_loss,
            "min_loss": min(epoch_losses),
            "max_loss": max(epoch_losses),
            "epoch_time": epoch_time,
            "batches_count": len(epoch_losses),
            "lr": scheduler.get_last_lr()[0],
            "timestamp": time.time(),
            "total_time": time.time() - training_start_time
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(epoch_log) + '\n')

        print(f"=== Epoch {epoch} завершена. Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s ===")

        # Сохранение лучшей модели
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0

            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': config.__dict__,
                'tokenizer_path': 'data/out_txt/tokenizer.json',
                'training_time': time.time() - training_start_time
            }

            torch.save(checkpoint, 'checkpoints/best_model.pt')
            print(f" Сохранена лучшая модель с loss: {avg_loss:.4f}")

            test_generation(model, tokenizer, device)
        else:
            patience += 1
            print(f"Loss не улучшился. Patience: {patience}/{max_patience}")

        # Критерии остановки
        if avg_loss < 2.0:
            print(" Достигнут хороший loss < 2.0! Обучение завершено.")
            break

        if patience >= max_patience:
            print(" Обучение остановлено из-за отсутствия улучшений.")
            break

    # Финальная статистика
    total_training_time = time.time() - training_start_time
    final_log = {
        "type": "final",
        "total_epochs": epoch + 1,
        "total_steps": global_step,
        "best_loss": best_loss,
        "total_training_time": total_training_time,
        "timestamp": time.time()
    }

    with open(log_file, 'a') as f:
        f.write(json.dumps(final_log) + '\n')

    print(f"Обучение завершено! Общее время: {total_training_time / 3600:.2f} часов")
    print(f"Логи сохранены в: {log_file}")

    return model, tokenizer


def analyze_training_logs(log_file_path):
    """Простой анализ логов обучения"""

    batch_losses = []
    epoch_data = []

    with open(log_file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())

            if data['type'] == 'batch':
                batch_losses.append({
                    'step': data['global_step'],
                    'loss': data['loss'],
                    'epoch': data['epoch']
                })
            elif data['type'] == 'epoch':
                epoch_data.append(data)

    print(f"Всего эпох: {len(epoch_data)}")
    print(f"Всего шагов: {len(batch_losses)}")

    if epoch_data:
        best_epoch = min(epoch_data, key=lambda x: x['avg_loss'])
        print(f"Лучшая эпоха: {best_epoch['epoch']} с loss: {best_epoch['avg_loss']:.4f}")
        print(f"Общее время обучения: {epoch_data[-1]['total_time'] / 3600:.2f} часов")

    return batch_losses, epoch_data


def analyze_training_stats(run_id):
    """Функция для анализа статистики обучения"""

    batch_file = f'training_logs/batch_stats_{run_id}.csv'
    epoch_file = f'training_logs/epoch_stats_{run_id}.csv'
    metadata_file = f'training_logs/metadata_{run_id}.json'

    if not all([os.path.exists(f) for f in [batch_file, epoch_file, metadata_file]]):
        print("Не все файлы статистики найдены!")
        return

    # Загрузка данных
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    import pandas as pd
    batch_df = pd.read_csv(batch_file)
    epoch_df = pd.read_csv(epoch_file)

    print(f" Анализ обучения Run ID: {run_id}")
    print(f"Время обучения: {metadata['total_training_time']:.2f}с")
    print(f"Эпох: {metadata['total_epochs']}, Батчей: {metadata['total_batches']}")
    print(f"Лучший loss: {metadata['best_loss']:.4f}")
    print(f"Финальный loss: {metadata['final_loss']:.4f}")

    # Можно добавить графики с matplotlib
    return batch_df, epoch_df, metadata


def test_generation(model, tokenizer, device):
    """Тестовая генерация для проверки качества модели"""
    model.eval()
    with torch.no_grad():
        test_prompt = "<BOS><GENERAL>Привет"
        start_ids = tokenizer.encode(test_prompt).ids
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

        generated = model.generate(x, max_new_tokens=50, temperature=0.8, top_k=50)
        generated_text = tokenizer.decode(generated[0].tolist())

        print(f" Тест генерации: {generated_text}")
    model.train()



class GPTChatBot:
    """Удобный интерфейс для работы с обученной моделью"""

    def __init__(self, checkpoint_path='checkpoints/best_model.pt'):
        """Загружаем обученную модель"""
        print("Загружаем обученную модель...")

        # Загружаем checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Восстанавливаем конфигурацию
        config_dict = checkpoint['config']
        self.config = GPTConfig(**config_dict)

        # Создаем модель
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GPT(self.config).to(self.device)

        # Загружаем веса
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Загружаем токенизатор
        tokenizer_path = checkpoint['tokenizer_path']
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        print(f" Модель загружена! Loss при сохранении: {checkpoint['loss']:.4f}")
        print(f" Устройство: {self.device}")

    def generate_text(self, prompt, max_tokens=200, temperature=0.8, top_k=50):
        """Генерация текста по промпту"""

        # Добавляем специальные токены если их нет
        if not prompt.startswith('<BOS>'):
            prompt = f"<BOS><GENERAL>{prompt}"

        # Токенизация
        try:
            input_ids = self.tokenizer.encode(prompt).ids
        except Exception as e:
            print(f"Ошибка токенизации: {e}")
            return None

        # Проверяем длину
        if len(input_ids) >= self.config.block_size:
            print(f"Промпт слишком длинный! Макс: {self.config.block_size}, получено: {len(input_ids)}")
            input_ids = input_ids[-self.config.block_size + 50:]  # Обрезаем, оставляя место для генерации

        # Конвертируем в тензор
        x = torch.tensor(input_ids, dtype=torch.long, device=self.device)[None, ...]

        # Генерируем
        with torch.no_grad():
            generated = self.model.generate(
                x,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )

        # Декодируем
        generated_text = self.tokenizear.decode(generated[0].tolist())

        # Убираем исходный промпт из результата
        result = generated_text[len(prompt):]

        return result.strip()

    def chat(self):
        """Интерактивный чат с моделью"""
        print("\n GPT Чат-бот готов!")
        print("Команды: 'quit' - выход, 'help' - помощь")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n👤 Вы: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(" До свидания!")
                    break

                if user_input.lower() == 'help':
                    self.show_help()
                    continue

                if not user_input:
                    print("Введите что-нибудь!")
                    continue

                print("\n Генерирую ответ...")

                # Генерация с разными настройками для разных типов промптов
                if any(word in user_input.lower() for word in ['расскажи', 'опиши', 'объясни']):
                    # Для объяснений - более консервативная генерация
                    response = self.generate_text(user_input, max_tokens=300, temperature=0.7, top_k=40)
                else:
                    # Обычная генерация
                    response = self.generate_text(user_input, max_tokens=200, temperature=0.8, top_k=50)

                if response:
                    print(f" Бот: {response}")
                else:
                    print(" Ошибка генерации")

            except KeyboardInterrupt:
                print("\n Прервано пользователем!")
                break
            except Exception as e:
                print(f" Ошибка: {e}")

    def show_help(self):
        """Показать справку"""
        print("""
🔧 Доступные команды и советы:

Команды:
- 'quit', 'exit', 'q' - выйти из чата
- 'help' - показать эту справку

Специальные токены (добавляются автоматически):
- <BOS> - начало текста
- <GENERAL> - общий контент
        """)


# =============================================================================
# ЧАСТЬ 3: ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# =============================================================================

def example_usage():
    """Примеры различных способов использования модели"""

    # Загружаем модель
    bot = GPTChatBot('checkpoints/best_model.pt')

    # Пример 1: Простая генерация
    print("=" * 60)
    print("ПРИМЕР 1: Простая генерация")
    print("=" * 60)

    prompts = [
        "Привет, как дела?",
        "Расскажи о программировании",
        "Что такое искусственный интеллект?",
        "Напиши короткий стих"
    ]

    for prompt in prompts:
        print(f"\n Промпт: {prompt}")
        response = bot.generate_text(prompt, max_tokens=150)
        print(f" Ответ: {response}")
        print("-" * 40)

    # Пример 2: Генерация с разными параметрами
    print("\n" + "=" * 60)
    print("ПРИМЕР 2: Разные настройки генерации")
    print("=" * 60)

    prompt = "Расскажи о космосе"

    settings = [
        {"temperature": 0.3, "top_k": 20, "desc": "Консервативно"},
        {"temperature": 0.8, "top_k": 50, "desc": "Сбалансированно"},
        {"temperature": 1.2, "top_k": 100, "desc": "Креативно"}
    ]

    for setting in settings:
        print(f"\n {setting['desc']} (temp={setting['temperature']}, top_k={setting['top_k']})")
        response = bot.generate_text(
            prompt,
            max_tokens=100,
            temperature=setting['temperature'],
            top_k=setting['top_k']
        )
        print(f"📖 {response}")

    # Пример 3: Интерактивный чат
    print("\n" + "=" * 60)
    print("ПРИМЕР 3: Запуск интерактивного чата")
    print("=" * 60)

    # Раскомментируйте для запуска чата:
    # bot.chat()


# =============================================================================
# КАК ЗАПУСКАТЬ ВСЕ ЭТО
# =============================================================================

if __name__ == "__main__":
    print(" Выберите действие:")
    print("1 - Обучить модель")
    print("2 - Использовать обученную модель")
    print("3 - Интерактивный чат")
    print("4 - Примеры использования")

    choice = input("Ваш выбор (1-4): ").strip()

    if choice == "1":
        # Запуск обучения
        train_model_with_saving()

    elif choice == "2":
        # Загрузка и тестирование модели
        bot = GPTChatBot()

        while True:
            prompt = input("\nВведите промпт (или 'quit' для выхода): ")
            if prompt.lower() == 'quit':
                break

            response = bot.generate_text(prompt)
            print(f"\nОтвет: {response}")

    elif choice == "3":
        # Интерактивный чат
        bot = GPTChatBot()
        bot.chat()

    elif choice == "4":
        # Примеры
        example_usage()

    else:
        print("Неверный выбор!")