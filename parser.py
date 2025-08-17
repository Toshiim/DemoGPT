import os
os.environ['HF_DATASETS_CACHE'] = os.path.abspath('./HFcache')
import yaml
import re
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from razdel import sentenize
import time
import psutil

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_parser.log', encoding='utf-8')
    ]
)


class DatasetParser:
    """
    Парсер датасетов с HuggingFace с фильтрацией и очисткой текста
    для обучения русскоязычных трансформеров
    """

    def __init__(self, config_path: str):
        """
        Инициализация парсера

        Args:
            config_path: Путь к YAML файлу конфигурации
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.filtering_config = self.config['filtering']
        self.output_config = self.config['output']

        # Преобразуем строку split в список для каждого датасета
        for dataset_config in self.config['datasets']:
            if isinstance(dataset_config['split'], str):
                dataset_config['split'] = [s.strip() for s in dataset_config['split'].split(',')]

        # Статистика обработки
        self.stats = {
            'total_records': 0,
            'filtered_records': 0,
            'split_records': 0,
            'emoji_filtered': 0,
            'chars_filtered': 0,
            'length_filtered': 0,
            'final_records': 0
        }

        # Компиляция регулярных выражений для производительности
        self._compile_regex_patterns()

        # Создание выходной директории
        os.makedirs(os.path.dirname(self.output_config['file_path']), exist_ok=True)

    def _compile_regex_patterns(self):
        """Компиляция регулярных выражений для быстрой фильтрации"""

        # Паттерн для эмодзи (основные категории Unicode)
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"  # эмотиконы
            "\U0001F300-\U0001F5FF"  # символы и пиктограммы
            "\U0001F680-\U0001F6FF"  # транспорт и символы карты
            "\U0001F1E0-\U0001F1FF"  # флаги (iOS)
            "\U00002700-\U000027BF"  # дингбаты
            "\U0001F900-\U0001F9FF"  # дополнительные символы
            "\U00002600-\U000026FF"  # разные символы
            "\U0001F170-\U0001F251"  # enclosed characters
            "]+", re.UNICODE
        )
        self.emoji_pattern = emoji_pattern

        self.html_tag_pattern = re.compile(r'<[^>]+>')

        # Допустимые символы: кириллица, латиница, цифры, знаки пунктуации, пробелы
        if self.filtering_config['allowed_chars_only']:
            allowed_chars = re.compile(
                r'^[а-яё'  # русские буквы строчные
                r'А-ЯЁ'  # русские буквы заглавные
                r'a-zA-Z'  # латинские буквы
                r'0-9'  # цифры
                r'\s'  # пробелы, табы, переносы строк
                r'.,!?;:()\[\]{}"\'«»—–\-'  # основная пунктуация
                r'№%@#$&*+=/<>\\|~`^_'  # доп. символы
                r']*$', re.UNICODE
            )
            self.allowed_chars_pattern = allowed_chars
        else:
            self.allowed_chars_pattern = None

    def _contains_emoji(self, text: str) -> bool:
        """
        Проверка наличия эмодзи в тексте

        Args:
            text: Текст для проверки

        Returns:
            True если найдены эмодзи
        """
        return bool(self.emoji_pattern.search(text))

    def _has_allowed_chars_only(self, text: str) -> bool:
        """
        Проверка, содержит ли текст только допустимые символы

        Args:
            text: Текст для проверки

        Returns:
            True если все символы допустимы
        """
        if self.allowed_chars_pattern is None:
            return True
        return bool(self.allowed_chars_pattern.match(text))

    def _is_text_valid(self, text: str) -> tuple[bool, str]:
        """
        Комплексная проверка валидности текста

        Args:
            text: Текст для проверки

        Returns:
            Tuple[bool, str]: (валиден ли текст, причина отклонения)
        """
        # Проверка длины
        if len(text) < self.filtering_config['min_length']:
            return False, "too_short"

        # Проверка на эмодзи
        if self.filtering_config['remove_emoji'] and self._contains_emoji(text):
            return False, "contains_emoji"

        # Проверка допустимых символов
        if not self._has_allowed_chars_only(text):
            return False, "invalid_chars"

        return True, "valid"

    def _split_long_text(self, text: str) -> List[str]:
        """
        Разбивка длинного текста на части по предложениям

        Args:
            text: Исходный текст

        Returns:
            Список частей текста
        """
        max_length = self.filtering_config['max_length']
        chunk_size = self.filtering_config['chunk_size']
        chunk_overlap = self.filtering_config['chunk_overlap']

        if len(text) <= max_length:
            return [text]

        # Разбиваем на предложения
        try:
            sentences = [s.text for s in sentenize(text)]
        except Exception as e:
            logging.error(f"ОШИБКА ДЕЛЕНИЯ ЧАНКОВ: {e}")
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Если добавление предложения не превысит chunk_size
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
            else:
                # Сохраняем текущий чанк если он достаточно длинный
                if current_chunk and len(current_chunk) >= self.filtering_config['min_length']:
                    chunks.append(current_chunk.strip())

                # Начинаем новый чанк
                # Если предложение само по себе длинное, обрезаем его
                if len(sentence) > chunk_size:
                    sentence = sentence[:chunk_size].rsplit(' ', 1)[0] + '.'

                # Добавляем перекрытие с предыдущим чанком
                if chunks and chunk_overlap > 0:
                    prev_words = chunks[-1].split()[-chunk_overlap // 6:]  # примерно 6 символов на слово
                    current_chunk = " ".join(prev_words) + " " + sentence
                else:
                    current_chunk = sentence

        # Добавляем последний чанк
        if current_chunk and len(current_chunk) >= self.filtering_config['min_length']:
            chunks.append(current_chunk.strip())

        return chunks

    def _process_record(self, record: Dict[str, Any], dataset_config: Dict[str, Any]) -> List[str]:
        """
        Обработка одной записи датасета

        Args:
            record: Запись из датасета
            dataset_config: Конфигурация датасета

        Returns:
            Список обработанных текстов
        """
        try:
            # Извлекаем текст
            text = record[dataset_config['text_column']]
            if not isinstance(text, str):
                return []

            text = self.html_tag_pattern.sub('', text)

            text = text.strip()
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                return []

            self.stats['total_records'] += 1

            # Проверяем валидность
            is_valid, reason = self._is_text_valid(text)
            if not is_valid:
                self.stats['filtered_records'] += 1
                if reason == "contains_emoji":
                    self.stats['emoji_filtered'] += 1
                elif reason == "invalid_chars":
                    self.stats['chars_filtered'] += 1
                elif reason == "too_short":
                    self.stats['length_filtered'] += 1
                return []

            # Разбиваем длинные тексты
            if len(text) > self.filtering_config['max_length']:
                chunks = self._split_long_text(text)
                self.stats['split_records'] += len(chunks) - 1
            else:
                chunks = [text]

            # Добавляем тег к каждому чанку
            tag = dataset_config['tag']
            processed_chunks = []
            for chunk in chunks:
                # Проверяем каждый чанк повторно (после разбивки)
                is_valid, _ = self._is_text_valid(chunk)
                if is_valid:
                    processed_text = f"{tag} {chunk.strip()}"
                    processed_chunks.append(processed_text)
                    self.stats['final_records'] += 1

            return processed_chunks

        except Exception as e:
            logging.warning(f"Ошибка при обработке записи: {e}")
            return []

    def _load_and_process_dataset(self, dataset_config: Dict[str, Any]) -> None:
        """
        Загрузка и обработка одного датасета
        """
        try:
            logging.info(f"Загружаем датасет: {dataset_config['name']}")

            # Открываем выходной файл один раз для всех сплитов
            with open(self.output_config['file_path'], 'a', encoding='utf-8', buffering=8192) as f:
                # Обрабатываем каждый сплит
                for split in dataset_config['split']:
                    logging.info(f"Загружаем сплит: {split}")
                    # Загружаем датасет для текущего сплита
                    if dataset_config['subset']:
                        dataset = load_dataset(
                            dataset_config['name'],
                            dataset_config['subset'],
                            split=split
                        )
                    else:
                        dataset = load_dataset(
                            dataset_config['name'],
                            split=split
                        )

                    logging.info(f"Загружено {len(dataset)} записей из {dataset_config['name']} (сплит: {split})")

                    # Обрабатываем записи
                    for record in tqdm(dataset, desc=f"Обработка {dataset_config['name']} ({split})"):
                        processed_texts = self._process_record(record, dataset_config)
                        for text in processed_texts:
                            f.write(text + '\n')

                    logging.info(f"Завершена обработка сплита {split} датасета {dataset_config['name']}")

        except Exception as e:
            logging.error(f"Ошибка при обработке датасета {dataset_config['name']}: {e}")
            # Обрабатываем записи
            with open(self.output_config['file_path'], 'a', encoding='utf-8', buffering=8192) as f:
                for record in tqdm(dataset, desc=f"Обработка {dataset_config['name']}"):
                    processed_texts = self._process_record(record, dataset_config)
                    for text in processed_texts:
                        f.write(text + '\n')

            logging.info(f"Завершена обработка {dataset_config['name']}")

        except Exception as e:
            logging.error(f"Ошибка при обработке датасета {dataset_config['name']}: {e}")

    def process_all_datasets(self) -> None:
        """
        Обработка всех датасетов из конфигурации
        """
        start_time = time.time()
        process = psutil.Process()

        logging.info("Начинаем обработку датасетов...")
        logging.info(f"Конфигурация фильтрации: {self.filtering_config}")

        # Очищаем выходной файл
        if os.path.exists(self.output_config['file_path']):
            os.remove(self.output_config['file_path'])

        # Обрабатываем каждый датасет
        for dataset_config in tqdm(self.config['datasets'], desc="Датасеты"):
            self._load_and_process_dataset(dataset_config)

        # Выводим статистику
        total_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 ** 2

        logging.info("=" * 50)
        logging.info("ИТОГОВАЯ СТАТИСТИКА")
        logging.info("=" * 50)
        logging.info(f"Время обработки: {total_time:.2f} сек ({total_time / 60:.2f} мин)")
        logging.info(f"Использование памяти: {final_memory:.2f} МБ")
        logging.info(f"Всего записей обработано: {self.stats['total_records']}")
        logging.info(f"Записей отфильтровано: {self.stats['filtered_records']}")
        logging.info(f"  - из-за эмодзи: {self.stats['emoji_filtered']}")
        logging.info(f"  - из-за недопустимых символов: {self.stats['chars_filtered']}")
        logging.info(f"  - из-за длины: {self.stats['length_filtered']}")
        logging.info(f"Записей разделено: {self.stats['split_records']}")
        logging.info(f"Финальных записей: {self.stats['final_records']}")

        if self.stats['total_records'] > 0:
            filter_rate = (self.stats['filtered_records'] / self.stats['total_records']) * 100
            logging.info(f"Процент фильтрации: {filter_rate:.2f}%")

        if os.path.exists(self.output_config['file_path']):
            file_size = os.path.getsize(self.output_config['file_path']) / 1024 ** 2
            logging.info(f"Размер выходного файла: {file_size:.2f} МБ")

        logging.info("=" * 50)
        logging.info(f"✅ Готово! Данные сохранены в: {self.output_config['file_path']}")


def main():

    config_path = "dataset_config.yaml"

    if not os.path.exists(config_path):
        logging.error(f"Файл конфигурации {config_path} не найден!")
        logging.info("Создайте файл dataset_config.yaml согласно примеру")
        return


    try:
        parser = DatasetParser(config_path)
        parser.process_all_datasets()
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()