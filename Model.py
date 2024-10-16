import numpy as np
import os
import glob
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Bidirectional, LSTM, TimeDistributed, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


EPOCH = 15
LEN_SEQ = 1000


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


# Функция для загрузки датасета
def load_dataset(dataset_folder):
    features_files = sorted(glob.glob(os.path.join(dataset_folder, 'features_*.npy')))
    labels_files = sorted(glob.glob(os.path.join(dataset_folder, 'labels_*.npy')))

    features_list = []
    labels_list = []

    for features_file, labels_file in zip(features_files, labels_files):
        features = np.load(features_file)
        labels = np.load(labels_file)

        features_list.append(features)
        labels_list.append(labels)

    return features_list, labels_list


# Функция для паддинга последовательностей признаков
def pad_sequences(sequences, maxlen):
    num_sequences = len(sequences)
    num_features = sequences[0].shape[1]
    padded_sequences = np.zeros((num_sequences, maxlen, num_features))
    for i, seq in enumerate(sequences):
        length = min(seq.shape[0], maxlen)
        padded_sequences[i, :length, :] = seq[:length, :]
    return padded_sequences


# Функция для паддинга меток
def pad_labels(labels_list, maxlen):
    num_sequences = len(labels_list)
    padded_labels = np.zeros((num_sequences, maxlen))
    for i, labels in enumerate(labels_list):
        length = min(len(labels), maxlen)
        padded_labels[i, :length] = labels[:length]
    return padded_labels


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
        val_pred = self.model.predict(val_data)
        val_pred_binary = (val_pred > 0.5).astype(int).flatten()
        val_true = val_labels.flatten()
        f1 = f1_score(val_true, val_pred_binary, zero_division=1)
        precision = precision_score(val_true, val_pred_binary, zero_division=1)
        recall = recall_score(val_true, val_pred_binary, zero_division=1)
        auc = roc_auc_score(val_true, val_pred.flatten())
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
        self.batch_bar = tqdm(total=self.params['steps'], desc=f"Epoch {epoch + 1}/{self.total_epochs}", position=1, leave=False)

    def on_batch_end(self, batch, logs=None):
        self.batch_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.update(1)
        self.batch_bar.close()

    def on_train_end(self, logs=None):
        self.epoch_bar.close()


# Основной блок
if __name__ == "__main__":
    # Укажите путь к папке с датасетом
    dataset_folder = r'big_dataset'  # Замените на ваш путь

    # Загружаем данные
    features_list, labels_list = load_dataset(dataset_folder)

    # Установка новой длины последовательности
    max_seq_length = LEN_SEQ

    # Паддим признаки и метки до новой длины для обучающих и тестовых данных
    split_index = int(len(features_list) * 0.8)
    X_train = pad_sequences(features_list[:split_index], max_seq_length)
    y_train = pad_labels(labels_list[:split_index], max_seq_length)

    X_test = pad_sequences(features_list[split_index:], max_seq_length)
    y_test = pad_labels(labels_list[split_index:], max_seq_length)

    print("Форма X_train после обрезки/паддинга:", X_train.shape)
    print("Форма y_train после обрезки/паддинга:", y_train.shape)
    print("Форма X_test после обрезки/паддинга:", X_test.shape)
    print("Форма y_test после обрезки/паддинга:", y_test.shape)

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
    exit()
    # Определяем форму входных данных
    input_shape = (X_train.shape[1], X_train.shape[2])  # (длина_последовательности, количество_признаков)
    model = build_model(input_shape)

    # Вывод структуры модели
    model.summary()

    # Определяем обратный вызов для вычисления F1-score, Precision, Recall и AUC-ROC
    validation_data = (X_test, y_test)
    f1_callback = F1ScoreCallback(validation_data=validation_data)

    # Проверка доступности GPU
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU device not found')
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
    pos_weight = 2.0  # Задайте нужный вес для положительного класса
    model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy(pos_weight), metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # Обучение модели
    history = model.fit(
        X_train, y_train,
        epochs=EPOCH,
        batch_size=8,
        validation_data=validation_data,
        callbacks=[f1_callback, progress_bar]
    )

    # Предсказание на тестовых данных
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Получение таймкодов
    timecodes = get_timecodes(y_pred_binary)

    # Вывод таймкодов для первого фрагмента
    for tc in timecodes:
        if tc['sample_idx'] == 0:
            print(f"Фрагмент {tc['sample_idx']}: {tc['start_time']:.2f} s - {tc['end_time']:.2f} s")

    # Сохранение модели
    model.save('notification_detection_model_with_weighted_bce_and_adamw.h5')

    # Визуализация потерь и точности
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Потери на обучении и валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['auc'], label='Train AUC-ROC')
    plt.plot(history.history['val_auc'], label='Validation AUC-ROC')
    plt.title('Точность и AUC-ROC на обучении и валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    plt.legend()

    plt.show()

    # Матрица ошибок (Confusion Matrix)
    y_pred_flat = y_pred_binary.flatten()
    y_true_flat = y_test.flatten()
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Предсказано')
    plt.ylabel('Истинно')
    plt.title('Матрица ошибок')
    plt.show()
