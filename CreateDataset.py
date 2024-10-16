import os
import re
import numpy as np
import librosa
from tqdm import tqdm


def parse_timecodes(text_file_path):
    """
    Парсит таймкоды из текстового файла, сдвигает их на 0.5 секунды назад и устанавливает длительность 2 секунды.

    Args:
        text_file_path (str): Путь к текстовому файлу.

    Returns:
        List[Tuple[float, float]]: Список кортежей (start_time, end_time) в секундах.
    """
    timecodes = []
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Удаляем метку "ДРУГОЕ", если она есть
        line = line.replace("ДРУГОЕ", "").strip()
        if not line:
            continue

        # Парсим время в формате ЧЧ:ММ:СС
        match = re.match(r'(\d{2}):(\d{2}):(\d{2})', line)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            total_seconds = hours * 3600 + minutes * 60 + seconds
            # Сдвигаем начало на 0.5 секунды назад
            start_time = total_seconds - 0.5
            # Устанавливаем длительность уведомления в 2 секунды
            end_time = start_time + 2
            # Проверка на отрицательное время
            if start_time < 0:
                start_time = 0
                end_time = 2
            timecodes.append((start_time, end_time))
    return timecodes


def create_regex_from_base_name(base_name):
    """
    Создает регулярное выражение из базового имени файла, заменяя подчеркивания на пробелы или подчеркивания.

    Args:
        base_name (str): Базовое имя аудиофайла без расширения.

    Returns:
        str: Регулярное выражение.
    """
    # Экранируем специальные символы
    escaped = re.escape(base_name)
    # Заменяем экранированные подчеркивания на шаблон, допускающий пробелы или подчеркивания
    pattern = escaped.replace(r'\_', r'[ _]+')
    return pattern


def find_text_files(audio_file, text_folder):
    """
    Находит все текстовые файлы с уведомлениями, соответствующие аудиофайлу.

    Args:
        audio_file (str): Имя аудиофайла.
        text_folder (str): Папка с текстовыми файлами.

    Returns:
        List[str]: Список путей к соответствующим текстовым файлам.
    """
    base_name = os.path.splitext(audio_file)[0]
    regex_pattern = create_regex_from_base_name(base_name) + r'(?:\s*\(Уведомления\))?\.txt$'
    pattern = re.compile(regex_pattern, re.IGNORECASE)

    matched_files = [f for f in os.listdir(text_folder) if pattern.match(f)]

    if not matched_files:
        print(f"Не найдено текстовых файлов для '{audio_file}'.")

    return [os.path.join(text_folder, f) for f in matched_files]


def normalize_features(features):
    """
    Нормализует спектрограмму по каждому частотному бинну.

    Args:
        features (np.ndarray): Спектрограмма (n_freq_bins, n_frames).

    Returns:
        np.ndarray: Нормализованная спектрограмма.
    """
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True) + 1e-6  # Добавляем epsilon для избежания деления на ноль
    normalized = (features - mean) / std
    return normalized


def split_audio_and_create_labels(audio_file_path, timecodes, segment_length=60, sr=16000,
                                  n_fft=1024, hop_length=512, window='hann'):
    """
    Разбивает аудио на фрагменты по 1 минуте, извлекает STFT признаки с 512 частотными биннами и создает метки для каждого фрагмента.

    Args:
        audio_file_path (str): Путь к аудиофайлу.
        timecodes (List[Tuple[float, float]]): Список интервалов уведомлений.
        segment_length (int): Длина фрагмента в секундах.
        sr (int): Частота дискретизации для загрузки аудио.
        n_fft (int): Размер FFT окна.
        hop_length (int): Шаг окна в отсчётах.
        window (str): Тип оконной функции.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, bool]]: Список кортежей (признаки, метки, содержит ли уведомление) для каждого фрагмента.
    """
    try:
        y, sr = librosa.load(audio_file_path, sr=sr)
    except Exception as e:
        print(f"Ошибка загрузки аудиофайла {audio_file_path}: {e}")
        return []

    total_duration = librosa.get_duration(y=y, sr=sr)

    num_segments = int(total_duration // segment_length) + 1

    features_labels = []

    for i in range(num_segments):
        segment_start = i * segment_length
        segment_end = min((i + 1) * segment_length, total_duration)

        # Извлекаем сегмент аудио
        start_sample = int(segment_start * sr)
        end_sample = int(segment_end * sr)
        y_segment = y[start_sample:end_sample]

        # Извлекаем STFT признаки
        stft = librosa.stft(y_segment, n_fft=n_fft, hop_length=hop_length, window=window)
        stft_magnitude = np.abs(stft)
        stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)

        # Отбрасываем последний частотный бин для получения 512 биннов
        stft_db = stft_db[:-1, :]  # Теперь форма (512, n_frames)

        # Временные метки
        times = librosa.frames_to_time(np.arange(stft_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)

        # Создаем метки
        labels = np.zeros(len(times), dtype=int)
        for notif_start, notif_end in timecodes:
            # Оптимизация: Используем булевую маску для всех фреймов
            mask = (segment_start + times >= notif_start) & (segment_start + times <= notif_end)
            labels = np.maximum(labels, mask.astype(int))

        # Проверяем, содержит ли фрагмент уведомление
        contains_notification = np.any(labels == 1)

        # Нормализуем спектрограмму
        stft_db_normalized = normalize_features(stft_db)  # (512, n_frames)

        # Транспонируем для соответствия формату (n_frames, n_freq_bins)
        stft_db_normalized = stft_db_normalized.T  # (n_frames, 512)

        features_labels.append((stft_db_normalized, labels, contains_notification))

    return features_labels


def process_all_files(audio_folder, text_folder, sr=16000, n_fft=1024, hop_length=512, window='hann'):
    """
    Обрабатывает все аудиофайлы и соответствующие текстовые файлы в указанных папках, используя STFT с 512 частотными биннами.

    Args:
        audio_folder (str): Папка с аудиофайлами.
        text_folder (str): Папка с текстовыми файлами.
        sr (int): Частота дискретизации для загрузки аудио.
        n_fft (int): Размер FFT окна.
        hop_length (int): Шаг окна в отсчётах.
        window (str): Тип оконной функции.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, bool]]: Список кортежей (признаки, метки, содержит ли уведомление) для всех фрагментов.
    """
    dataset = []
    audio_files = [f for f in os.listdir(audio_folder) if f.lower().endswith(('.mp3', '.wav'))]

    for audio_file in tqdm(audio_files, desc="Обработка файлов"):
        audio_file_path = os.path.join(audio_folder, audio_file)
        text_file_paths = find_text_files(audio_file, text_folder)

        if not text_file_paths:
            continue  # Сообщение уже выводится в find_text_files
        # Парсим все таймкоды из найденных текстовых файлов
        all_timecodes = []
        for text_file_path in text_file_paths:
            try:
                timecodes = parse_timecodes(text_file_path)
                all_timecodes.extend(timecodes)
            except Exception as e:
                print(f"Ошибка при парсинге файла {text_file_path}: {e}")

        # Разбиваем аудио и извлекаем признаки и метки
        features_labels = split_audio_and_create_labels(audio_file_path, all_timecodes,
                                                        sr=sr, n_fft=n_fft,
                                                        hop_length=hop_length, window=window)

        dataset.extend(features_labels)

    return dataset


def save_dataset(dataset, output_folder='dataset'):
    """
    Сохраняет датасет в указанный каталог.

    Args:
        dataset (List[Tuple[np.ndarray, np.ndarray, bool]]): Список кортежей (признаки, метки, содержит ли уведомление).
        output_folder (str): Папка для сохранения датасета.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    for idx, (features, labels, _) in enumerate(dataset):
        features_path = os.path.join(output_folder, f'features_{idx}.npy')
        labels_path = os.path.join(output_folder, f'labels_{idx}.npy')
        np.save(features_path, features)
        np.save(labels_path, labels)

    print(f"Датасет сохранен в папке {output_folder}")


def count_labels(dataset):
    """
    Подсчитывает количество меток с 1 и с 0 в датасете.

    Args:
        dataset (List[Tuple[np.ndarray, np.ndarray, bool]]): Список кортежей (признаки, метки, содержит ли уведомление).

    Returns:
        Tuple[int, int]: Количество меток с 1 и с 0 соответственно.
    """
    total_ones = sum(np.sum(labels) for _, labels, _ in dataset)
    total_zeros = sum(len(labels) - np.sum(labels) for _, labels, _ in dataset)
    return total_ones, total_zeros


# Пример использования
if __name__ == "__main__":
    audio_folder = r"D:\PyCharmProjects\Sounded\data"  # Замените на путь к папке с аудиофайлами
    text_folder = r"D:\PyCharmProjects\Sounded\data"  # Замените на путь к папке с текстовыми файлами
    output_folder = 'test_dataset'  # Папка для сохранения датасета

    # Определите параметры STFT
    sr = 16000
    n_fft = 1024  # Установлено на 1024 для получения 512 частотных биннов
    hop_length = 512
    window = 'hann'

    # Обрабатываем все файлы и получаем исходный датасет
    dataset = process_all_files(audio_folder, text_folder, sr=sr, n_fft=n_fft,
                                hop_length=hop_length, window=window)

    # Подсчитываем метки
    ones, zeros = count_labels(dataset)
    print(f"Количество меток с 1: {ones}")
    print(f"Количество меток с 0: {zeros}")

    # Сохраняем датасет
    save_dataset(dataset, output_folder)
