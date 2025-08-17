import os
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import re
import random
import psutil
import logging
import time
from itertools import islice
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


input_path = "data/parsed_dataset.txt"
output_dir = "data/out_txt"
max_lines = 721241    # Первые n строк для токенизации
vocab_size = 10000
special_tokens = ["<BOS>", "<EOS>", "<PAD>", "<UNK>", "<GENERAL>", "<FRAG_START>", "<FRAG_END>"]
max_length = 256 # длина батча
min_stride = 128
max_stride = 230
num_workers = max(1, os.cpu_count() or 4)
max_chunk_lines = 2000 # Количество строк в чанке
tokenizer_train_lines = 100000  # Строки для обучения токенизатора
random_seed = 42
sample_limit = 721241    # Ограничение для выборки обучения токенизатора


def analyze_file(filepath: str):
    import os

    if not os.path.exists(filepath):
        print(f"Файл не найден: {filepath}")
        return

    file_size = os.path.getsize(filepath)

    line_count = 0
    total_length = 0
    unique_chars = set()

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            line_count += 1
            total_length += len(line)
            unique_chars.update(line)

    avg_length = total_length / line_count if line_count else 0

    print(f"Путь к файлу: {filepath}")
    print(f"Размер файла: {file_size / 1024:.2f} КБ")
    print(f"Количество строк: {line_count}")
    print(f"Средняя длина строки: {avg_length:.2f}")
    print(f"Количество уникальных символов: {len(unique_chars)}")
    print(f"Уникальные символы: {''.join(sorted(unique_chars))}")


def reservoir_sampling(file_path, sample_size, sample_limit):
    random.seed(random_seed)
    reservoir = []
    start_time = time.time()
    try:
        with open(file_path, "r", encoding="utf-8", buffering=8192) as f:
            for i, line in enumerate(tqdm(islice(f, sample_limit), desc="Случайная выборка строк", total=sample_limit)):
                if not line.strip():
                    logging.warning(f"Пропущена пустая строка {i + 1}")
                    continue
                line = re.sub(r'(<\w+>\s+)', r'\1', line)
                if i < sample_size:
                    reservoir.append(line)
                else:
                    if random.random() < sample_size / (i + 1):
                        reservoir[random.randint(0, sample_size - 1)] = line
        logging.info(f"Выбрано {len(reservoir)} строк за {time.time() - start_time:.2f} сек")
        return reservoir
    except Exception as e:
        logging.error(f"Ошибка при случайной выборке: {e}")
        return []



def train_tokenizer(lines):
    try:
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=3,
            initial_alphabet=list("!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuv"
                                  "wxyz{|}~«»ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё–—№")
        )

        temp_file = os.path.join(output_dir, "temp_train.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        tokenizer.train(files=[temp_file], trainer=trainer)
        os.remove(temp_file)
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        tokenizer.save(tokenizer_path)
        logging.info(f"Токенизатор сохранён: {tokenizer_path}")
        return tokenizer_path
    except Exception as e:
        logging.error(f"Ошибка при обучении токенизатора: {e}")
        raise


def extract_tag(record):
    try:
        if not isinstance(record, str):
            logging.error(f"Некорректный тип записи: {type(record)}, содержимое: {record}")
            return None, ""
        match = re.match(r'^<(\w+)>', record)
        if match:
            tag = match.group(0)
            text = record[len(tag):].strip()
            return tag, text
        return None, record.strip()
    except Exception as e:
        logging.error(f"Ошибка при извлечении тега: {e}")
        return None, ""

def tokenize_record(record, tokenizer_path, max_length, min_stride, max_stride):
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tag, text = extract_tag(record)
        input_text = f"<BOS>{tag or ''}{text}<EOS>"
        encoded = tokenizer.encode(input_text)
        tokens = encoded.ids
        fragments = []
        i = 0
        while i < len(tokens):
            chunk = tokens[i:i + max_length]
            if i == 0:
                if len(chunk) < max_length:
                    chunk += [tokenizer.token_to_id("<PAD>")] * (max_length - len(chunk))
                fragments.append(chunk)
            elif i + max_length >= len(tokens):
                if len(chunk) < max_length:
                    chunk += [tokenizer.token_to_id("<PAD>")] * (max_length - len(chunk))
                fragments.append(chunk)
            else:
                chunk = [tokenizer.token_to_id("<FRAG_START>")] + (
                    [tokenizer.token_to_id(tag)] if tag else []
                ) + chunk[:-1] + [tokenizer.token_to_id("<FRAG_END>")]
                if len(chunk) < max_length:
                    chunk += [tokenizer.token_to_id("<PAD>")] * (max_length - len(chunk))
                elif len(chunk) > max_length:
                    chunk = chunk[:max_length]
                fragments.append(chunk)
            i += random.randint(min_stride, max_stride)
        return fragments
    except Exception as e:
        logging.error(f"Ошибка при токенизации записи: {e}")
        return []


def tokenize_chunk(chunk_lines, tokenizer_path, max_length, min_stride, max_stride, chunk_id):
    try:
        all_fragments = []
        for i, line in enumerate(tqdm(chunk_lines, desc=f"Токенизация чанка {chunk_id}")):
            fragments = tokenize_record(line, tokenizer_path, max_length, min_stride, max_stride)
            if fragments:
                all_fragments.extend(fragments)
            else:
                logging.warning(f"Пустой список фрагментов для строки {i + 1} в чанке {chunk_id}")
        if not all_fragments:
            logging.error(f"Чанк {chunk_id} пустой")
            return 0
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_id}.npy")
        fragments_array = np.array(all_fragments, dtype=np.int32)
        np.save(chunk_path, fragments_array)
        logging.info(f"Сохранён чанк {chunk_id}, фрагментов: {len(all_fragments)}")
        return len(all_fragments)
    except Exception as e:
        logging.error(f"Ошибка при токенизации чанка {chunk_id}: {e}")
        return 0

def stream_chunks(file_path, max_lines, max_chunk_lines):
    try:
        with open(file_path, "r", encoding="utf-8", buffering=8192) as f:
            chunk = []
            line_count = 0
            for line in tqdm(islice(f, max_lines), desc="Чтение чанков", total=max_lines):
                if not line.strip():
                    logging.warning(f"Пропущена пустая строка {line_count + 1}")
                    continue
                line = re.sub(r'(<\w+>\s+)', r'\1', line)
                chunk.append(line)
                line_count += 1
                if len(chunk) >= max_chunk_lines:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
    except Exception as e:
        logging.error(f"Ошибка при стриминге чанков: {e}")
        return []

def main():
    #analyze_file(input_path)
    start_time = time.time()
    process = psutil.Process()
    peak_memory = 0
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Папка {output_dir} создана или уже существует")
    if not os.path.exists(input_path):
        logging.error(f"Файл {input_path} не найден")
        return
    mem_info = process.memory_info()
    logging.info(f"Начальное использование памяти: {mem_info.rss / 1024**2:.2f} МБ")

    # Случайная выборка строк
    tokenizer_lines = reservoir_sampling(input_path, tokenizer_train_lines, sample_limit)
    if not tokenizer_lines:
        logging.error("Не удалось выбрать строки для токенизатора")
        return

    # Обучение токенизатора
    try:
        tokenizer_path = train_tokenizer(tokenizer_lines)
    except Exception as e:
        logging.error(f"Не удалось обучить токенизатор: {e}")
        return

    # Стриминг и токенизация чанков
    total_fragments = 0
    chunk_id = 0
    futures = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk in stream_chunks(input_path, max_lines, max_chunk_lines):
            if chunk:
                # Связываем future с его chunk_id
                future = executor.submit(tokenize_chunk, chunk, tokenizer_path, max_length, min_stride, max_stride,
                                         chunk_id)
                futures[future] = chunk_id
                chunk_id += 1

    # Сбор результатов ПОСЛЕ завершения всех задач
    results = {}
    logging.info("Сбор результатов от дочерних процессов...")
    for future in tqdm(as_completed(futures), total=len(futures), desc="Сбор результатов"):
        cid = futures[future]
        try:
            num_fragments_in_chunk = future.result()
            if num_fragments_in_chunk > 0:
                results[cid] = num_fragments_in_chunk
            else:
                logging.warning(f"Чанк {cid} не дал фрагментов или произошла ошибка. Он будет пропущен.")
        except Exception as e:
            logging.error(f"Критическая ошибка в обработке чанка {cid}: {e}")

    if not results:
        logging.error("Не создано ни одного фрагмента. Прерывание.")
        return

    total_fragments = sum(results.values())
    logging.info(f"Точное общее число фрагментов для сохранения: {total_fragments}")

    final_shape = (total_fragments, max_length)
    bin_path = os.path.join(output_dir, "train_fragments.npy")
    final_memmap = np.memmap(bin_path, dtype=np.int64, mode='w+', shape=final_shape) # int32 -> int 64 для оптимизации .astype(np.int64) в обучении.

    offset = 0
    for i in tqdm(sorted(results.keys()), desc="Объединение чанков"):
        chunk_path = os.path.join(output_dir, f"chunk_{i}.npy")
        if os.path.exists(chunk_path):
            try:
                chunk_data = np.load(chunk_path)
                if len(chunk_data) == results[i]:
                    final_memmap[offset:offset + len(chunk_data)] = chunk_data
                    offset += len(chunk_data)
                    os.remove(chunk_path)
                else:
                    logging.warning(
                        f"Несоответствие размера для чанка {i}. Ожидалось {results[i]}, получено {len(chunk_data)}. Пропускаем.")
            except Exception as e:
                logging.error(f"Не удалось загрузить или объединить чанк {i}: {e}")
        else:
            logging.warning(f"Чанк {chunk_path} ожидался, но не найден на диске.")

    final_memmap.flush()
    logging.info(f"Финальный файл сохранён: {bin_path}, фрагментов: {offset}")

    if offset != total_fragments:
        logging.error(
            f"Финальное количество фрагментов ({offset}) не совпадает с расчетным ({total_fragments})! Файл может быть поврежден.")

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "vocab_size": vocab_size,
            "tokenizer": "custom_bpe",
            "num_fragments": total_fragments,
            "max_length": max_length,
            "min_stride": min_stride,
            "max_stride": max_stride,
            "num_lines": max_lines,
            "avg_words_per_line": 0.0
        }, f, indent=2)
    logging.info(f"Метаинформация сохранена: {meta_path}")

    total_time = time.time() - start_time
    file_size = os.path.getsize(bin_path) / 1024**3 if os.path.exists(bin_path) else 0
    avg_fragments_per_line = total_fragments / max_lines if max_lines > 0 else 0
    logging.info(f"===== Итоговая статистика =====")
    logging.info(f"Общее время выполнения: {total_time:.2f} сек ({total_time / 60:.2f} мин)")
    logging.info(f"Пиковое использование ОЗУ: {peak_memory / 1024**2:.2f} МБ")
    logging.info(f"Общее число фрагментов: {total_fragments}")
    logging.info(f"Среднее число фрагментов на строку: {avg_fragments_per_line:.2f}")
    logging.info(f"Размер файла train_fragments.npy: {file_size:.2f} ГБ")
    logging.info(f"Скорость обработки: {max_lines / total_time:.2f} строк/с")
    logging.info(f"==============================")
    logging.info(f"✅ Готово. Сохранено {total_fragments} фрагментов.")

if __name__ == "__main__":
    main()