import numpy as np
import os
import glob
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Bidirectional, LSTM, TimeDistributed, Dropout, \
    Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# Параметры
EPOCH = 15
LEN_SEQ = 1000  # Максимальная длина последовательности
BATCH_SIZE = 8


# Взвешенная бинарная кросс-энтропия
def weighted_binary_crossentropy(pos_weight):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weight = y_true * pos_weight + (1. - y_true)
        weighted_bce = weight * bce
        return tf.reduce_mean(weighted_bce)

    return loss


# Функция для получения таймкодов из бинарных предсказаний
def get_timecodes(y_pred_binary, sr=16000, frame_step=0.01):
    timecodes_list = []
    hop_length = int(sr * frame_step)

    for sample_idx in range(y_pred_binary.shape[0]):
        binary_sequence = y_pred_binary[sample_idx].flatten()
        indices = np.where(np.diff(binary_sequence) != 0)[0] + 1

        # Обработка случая, когда уведомление начинается или заканчивается на границе
        if binary_sequence[0] == 1:
            indices = np.insert(indices, 0, 0)
        if binary_sequence[-1] == 1:
            indices = np.append(indices, len(binary_sequence))

        # Группируем индексы по парам (start, end)
        if len(indices) % 2 != 0:
            indices = np.append(indices, len(binary_sequence))
        start_end_indices = indices.reshape(-1, 2)

        # Преобразуем индексы во время
        for start_idx, end_idx in start_end_indices:
            start_time = librosa.frames_to_time(start_idx, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(end_idx, sr=sr, hop_length=hop_length)
            timecodes_list.append({
                'sample_idx': sample_idx,
                'start_time': start_time,
                'end_time': end_time
            })
    return timecodes_list


# Функция для загрузки сбалансированного датасета
def load_balanced_dataset(dataset_folder):
    """
    Загружает сбалансированные фреймы признаков и меток из файлов .npy.

    Args:
        dataset_folder (str): Путь к папке с сохраненным датасетом.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Массивы фреймов признаков и меток.
    """
    features_path = os.path.join(dataset_folder, 'balanced_features.npy')
    labels_path = os.path.join(dataset_folder, 'balanced_labels.npy')

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Файл с признаками не найден: {features_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Файл с метками не найден: {labels_path}")

    features = np.load(features_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    return features, labels


# Функция для паддинга последовательностей признаков
def pad_sequences_custom(features, maxlen):
    """
    Паддинг последовательностей признаков до максимальной длины.

    Args:
        features (List[np.ndarray]): Список последовательностей фреймов признаков.
        maxlen (int): Максимальная длина последовательности.

    Returns:
        np.ndarray: Падженные последовательности признаков.
    """
    num_samples = len(features)
    num_features = features[0].shape[1]  # Assume shape (frames, num_features)
    padded_features = np.zeros((num_samples, maxlen, num_features))

    for i, feature in enumerate(features):
        length = min(feature.shape[0], maxlen)
        padded_features[i, :length, :] = feature[:length, :]

    return padded_features


# Функция для паддинга меток
def pad_labels_custom(labels, maxlen):
    """
    Паддинг меток до максимальной длины.

    Args:
        labels (List[np.ndarray]): Список последовательностей меток.
        maxlen (int): Максимальная длина последовательности.

    Returns:
        np.ndarray: Падженные метки.
    """
    num_samples = len(labels)
    padded_labels = np.zeros((num_samples, maxlen))

    for i, label in enumerate(labels):
        length = min(len(label), maxlen)
        padded_labels[i, :length] = label[:length]

    return padded_labels


# Функция для балансировки датасета на уровне последовательностей
def balance_dataset_sequential(features, labels, downsample_factor=12, oversample_factor=4, random_seed=42):
    """
    Балансирует датасет путем доунсемплинга последовательностей без уведомлений и оверсэмплинга с уведомлениями.

    Args:
        features (List[np.ndarray]): Список последовательностей фреймов признаков.
        labels (List[np.ndarray]): Список последовательностей меток.
        downsample_factor (int): Коэффициент доунсемплинга для последовательностей с метками 0.
        oversample_factor (int): Коэффициент оверсэмплинга для последовательностей с метками 1.
        random_seed (int): Случайное зерно для воспроизводимости.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Сбалансированные списки последовательностей признаков и меток.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Разделяем на последовательности с уведомлениями и без
    sequences_with_notification = []
    sequences_without_notification = []

    for f, l in zip(features, labels):
        if np.any(l == 1):
            sequences_with_notification.append((f, l))
        else:
            sequences_without_notification.append((f, l))

    # Доунсемплинг последовательностей без уведомлений
    num_sequences_without = len(sequences_without_notification) // downsample_factor
    if num_sequences_without > 0:
        sequences_without_downsampled = random.sample(sequences_without_notification, num_sequences_without)
    else:
        sequences_without_downsampled = []

    # Оверсэмплинг последовательностей с уведомлениями
    sequences_with_oversampled = sequences_with_notification * oversample_factor

    # Объединяем сбалансированные данные
    balanced_sequences = sequences_without_downsampled + sequences_with_oversampled

    # Перемешиваем данные
    random.shuffle(balanced_sequences)

    # Разделяем обратно на признаки и метки
    balanced_features = [seq[0] for seq in balanced_sequences]
    balanced_labels = [seq[1] for seq in balanced_sequences]

    return balanced_features, balanced_labels


# Функция для построения модели
def build_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv1D(512, kernel_size=9, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    x = Conv1D(256, kernel_size=9, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, kernel_size=9, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv1D(64, kernel_size=9, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv1D(32, kernel_size=9, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv1D(16, kernel_size=9, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = Dropout(0.3)(x)

    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    model = Model(inputs, outputs)

    return model


# Класс обратного вызова для вычисления F1-score, Precision, Recall и AUC-ROC на валидационной выборке
class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_data, val_labels = self.validation_data
        val_pred = self.model.predict(val_data, verbose=0)
        y_pred_binary = (val_pred > 0.5).astype(int).flatten()
        y_true = val_labels.flatten()
        f1 = f1_score(y_true, y_pred_binary, zero_division=1)
        precision = precision_score(y_true, y_pred_binary, zero_division=1)
        recall = recall_score(y_true, y_pred_binary, zero_division=1)
        try:
            auc = roc_auc_score(y_true, val_pred.flatten())
        except ValueError:
            auc = 0.0  # Если все метки одного класса, AUC не определен
        print(f' - val_f1: {f1:.4f} - val_precision: {precision:.4f} - val_recall: {recall:.4f} - val_auc: {auc:.4f}')


# Класс обратного вызова для прогресса с использованием tqdm
class TQDMProgressBar(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=self.total_epochs, desc="Epochs", position=0)

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_bar = tqdm(total=self.params['steps'], desc=f"Epoch {epoch + 1}/{self.total_epochs}", position=1,
                              leave=False)

    def on_batch_end(self, batch, logs=None):
        self.batch_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.update(1)
        self.batch_bar.close()

    def on_train_end(self, logs=None):
        self.epoch_bar.close()


# Основной блок
if __name__ == "__main__":
    # Укажите путь к папке с сбалансированным датасетом
    dataset_folder = r"D:\PyCharmProjects\Sounded\Tools\balanced_dataset"  # Замените на ваш путь

    # Загружаем сбалансированные данные
    print("Загрузка сбалансированного датасета...")
    all_features, all_labels = load_balanced_dataset(dataset_folder)
    print(f"Количество последовательностей признаков: {len(all_features)}")
    print(f"Количество последовательностей меток: {len(all_labels)}")

    # Балансировка данных
    print("Балансировка датасета...")
    balanced_features, balanced_labels = balance_dataset_sequential(all_features, all_labels)

    print(f"Количество сбалансированных последовательностей: {len(balanced_features)}")

    # Определяем максимальную длину последовательности
    max_seq_length = LEN_SEQ
    # Паддинг
    print("Паддинг последовательностей признаков...")
    X_padded = pad_sequences_custom(balanced_features, max_seq_length)
    print("Паддинг меток...")
    y_padded = pad_labels_custom(balanced_labels, max_seq_length)

    print("Форма X_padded:", X_padded.shape)
    print("Форма y_padded:", y_padded.shape)

    # Разделяем на обучающую и тестовую выборки (80% обучающая, 20% тестовая)
    split_index = int(len(X_padded) * 0.8)
    X_train, X_test = X_padded[:split_index], X_padded[split_index:]
    y_train, y_test = y_padded[:split_index], y_padded[split_index:]

    print("Форма X_train после паддинга:", X_train.shape)
    print("Форма y_train после паддинга:", y_train.shape)
    print("Форма X_test после паддинга:", X_test.shape)
    print("Форма y_test после паддинга:", y_test.shape)

    # Расширяем размерность меток для совместимости с моделью
    y_train = y_train[..., np.newaxis]
    y_test = y_test[..., np.newaxis]

    # Подсчет количества элементов в каждом классе для тренировочного набора
    train_class_0 = np.sum(y_train == 0)
    train_class_1 = np.sum(y_train == 1)

    # Подсчет количества элементов в каждом классе для тестового набора
    test_class_0 = np.sum(y_test == 0)
    test_class_1 = np.sum(y_test == 1)

    # Вывод результатов
    print(f"Количество элементов класса '0' в y_train: {train_class_0}")
    print(f"Количество элементов класса '1' в y_train: {train_class_1}")
    print(f"Количество элементов класса '0' в y_test: {test_class_0}")
    print(f"Количество элементов класса '1' в y_test: {test_class_1}")

    # Вычисление pos_weight на основе тренировочных данных
    pos_weight = train_class_0 / train_class_1 if train_class_1 != 0 else 1.0
    print(f"Вычисленный pos_weight: {pos_weight:.4f}")

    # Определяем форму входных данных
    input_shape = (X_train.shape[1], X_train.shape[2])  # (длина_последовательности, количество_признаков)

    # Строим модель
    model = build_model(input_shape)

    # Вывод структуры модели
    model.summary()

    # Определяем обратный вызов для вычисления F1-score, Precision, Recall и AUC-ROC
    validation_data = (X_test, y_test)
    f1_callback = F1ScoreCallback(validation_data=validation_data)

    # Проверка доступности GPU
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU device not found. Обучение будет происходить на CPU, что может быть медленнее.')
    else:
        print('Found GPU at: {}'.format(device_name))

    # Callback для прогресса
    progress_bar = TQDMProgressBar(total_epochs=EPOCH)

    # Вывод распределения меток
    print("Распределение меток в y_train:", np.unique(y_train, return_counts=True))
    print("Распределение меток в y_test:", np.unique(y_test, return_counts=True))

    # Оптимизатор AdamW с весовым распадом
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)

    # Компиляция модели с взвешенной бинарной кросс-энтропией
    model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy(pos_weight),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # Обучение модели
    print("Начало обучения модели...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        validation_data=validation_data,
        callbacks=[f1_callback, progress_bar]
    )

    # Предсказание на тестовых данных
    print("Выполнение предсказаний на тестовой выборке...")
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Получение таймкодов
    print("Получение таймкодов из предсказаний...")
    timecodes = get_timecodes(y_pred_binary)

    # Вывод таймкодов для первого фрагмента
    print("Таймкоды для первого фрагмента:")
    for tc in timecodes:
        if tc['sample_idx'] == 0:
            print(f"Фрагмент {tc['sample_idx']}: {tc['start_time']:.2f} s - {tc['end_time']:.2f} s")

    # Сохранение модели
    model_save_path = 'notification_detection_model_with_weighted_bce_and_adamw.h5'
    print(f"Сохранение модели в {model_save_path}...")
    model.save(model_save_path)

    # Визуализация потерь и точности
    print("Визуализация результатов обучения...")
    plt.figure(figsize=(12, 4))

    # Потери
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Потери на обучении и валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.legend()

    # Точность и AUC-ROC
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['auc'], label='Train AUC-ROC')
    plt.plot(history.history['val_auc'], label='Validation AUC-ROC')
    plt.title('Точность и AUC-ROC на обучении и валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Матрица ошибок (Confusion Matrix)
    print("Построение матрицы ошибок...")
    y_pred_flat = y_pred_binary.flatten()
    y_true_flat = y_test.flatten()
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['True 0', 'True 1'])
    plt.xlabel('Предсказано')
    plt.ylabel('Истинно')
    plt.title('Матрица ошибок')
    plt.show()
