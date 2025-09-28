import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
# >>> НОВОЕ: Добавляем confusion_matrix
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix

# --- 1. НАСТРОЙКИ (ИЗМЕНИТЕ ЭТИ ЗНАЧЕНИЯ) ---

# Укажите путь к папке с вашей сохраненной моделью
MODEL_PATH = "./back-front/models_4" # Пример: "./models/baseline_bert" или другая папка

# Укажите путь к вашему размеченному тестовому файлу
TEST_CSV_PATH = "profanity_dataset_500.csv" # Файл на 500 строк

# Названия колонок в вашем CSV
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Параметры, которые должны совпадать с параметрами обучения
MAX_LENGTH = 256
BATCH_SIZE = 16

# --- КОНЕЦ НАСТРОЕК ---


def evaluate_model(model_path, csv_path):
    """
    Загружает сохраненную модель, делает предсказания на CSV файле
    и выводит метрики, включая матрицу ошибок.
    """
    print("="*50)
    print(f"Запуск оценки для модели: {model_path}")
    print(f"Данные для оценки: {csv_path}")
    print("="*50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    try:
        print("Загрузка модели и токенизатора...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(device)
    except OSError:
        print(f"!!! ОШИБКА: Не удалось найти модель по пути: {model_path}")
        return

    print("Загрузка и подготовка данных для оценки...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"!!! ОШИБКА: Не удалось найти файл с данными: {csv_path}")
        return
        
    df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(
            examples[TEXT_COLUMN], padding="max_length", truncation=True, max_length=MAX_LENGTH
        )

    print("Токенизация данных...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    print("Запуск предсказания...")
    training_args = TrainingArguments(
        output_dir="./temp_eval_results", per_device_eval_batch_size=BATCH_SIZE, fp16=torch.cuda.is_available()
    )
    trainer = Trainer(model=model, args=training_args)
    
    raw_predictions = trainer.predict(tokenized_dataset)
    logits = raw_predictions.predictions
    predicted_labels = np.argmax(logits, axis=1)
    true_labels = df[LABEL_COLUMN].values

    # --- 6. Расчет и вывод метрик ---
    print("\n--- РЕЗУЛЬТАТЫ ОЦЕНКИ ---")

    # >>> НОВОЕ: Вывод матрицы ошибок (Confusion Matrix) <<<
    print("\n--- Матрица ошибок (TP/FP/TN/FN) ---")
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Извлекаем значения TN, FP, FN, TP
    # Матрица имеет вид: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    # Рисуем красивую таблицу
    print(f"               | Предсказано: НЕТ | Предсказано: ЕСТЬ")
    print(f"-----------------------------------------------------")
    print(f" Реально: НЕТ    | {tn:^16} | {fp:^16}")
    print(f" Реально: ЕСТЬ   | {fn:^16} | {tp:^16}")
    print(f"-----------------------------------------------------")

    print("\nРасшифровка:")
    print(f"  - True Negative  (TN): {tn:<5} | Модель верно определила, что мата НЕТ.")
    print(f"  - False Positive (FP): {fp:<5} | Ошибка! Модель нашла мат там, где его НЕТ (Ложная тревога).")
    print(f"  - False Negative (FN): {fn:<5} | Ошибка! Модель НЕ нашла мат там, где он ЕСТЬ (Пропуск).")
    print(f"  - True Positive  (TP): {tp:<5} | Модель верно определила, что мат ЕСТЬ.")
    
    # Главная метрика для соревнования
    f1_binary = f1_score(true_labels, predicted_labels, average='binary')
    print(f"\n🎯 Главная метрика (F1-score для класса 1): {f1_binary:.4f}")
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"   Общая точность (Accuracy): {accuracy:.4f}\n")
    
    # Подробный отчет по всем метрикам
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=['Класс 0 (нет мата)', 'Класс 1 (есть мат)']
    )
    print("Подробный отчет:")
    print(report)
    print("="*50 + "\n")


if __name__ == '__main__':
    evaluate_model(MODEL_PATH, TEST_CSV_PATH)