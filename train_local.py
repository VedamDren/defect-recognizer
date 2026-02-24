import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import numpy as np

# ------------------------------
# 1. Параметры (можно менять)
# ------------------------------
IMG_SIZE = 224          # размер изображений, на котором обучалась предобученная сеть
BATCH_SIZE = 32         # количество изображений за один шаг (уменьшите, если мало памяти)
EPOCHS = 20             # количество эпох обучения
NUM_CLASSES = 2
CLASS_NAMES = ['punching_hole', 'rolled_pit']

# Пути к данным (используем относительные папки внутри проекта)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # папка проекта
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')         # E:\...\defect-recognizer\dataset\train
VAL_DIR   = os.path.join(BASE_DIR, 'dataset', 'validation')    # E:\...\defect-recognizer\dataset\validation

# ------------------------------
# 2. Проверка наличия папок
# ------------------------------
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Папка с обучающими данными не найдена: {TRAIN_DIR}")
if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"Папка с валидационными данными не найдена: {VAL_DIR}")

# ------------------------------
# 3. Генераторы данных с аугментацией
# ------------------------------
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,                 # нормализация пикселей
    rotation_range=20,               # случайный поворот
    width_shift_range=0.2,           # сдвиг по ширине
    height_shift_range=0.2,          # сдвиг по высоте
    shear_range=0.2,                 # сдвиг
    zoom_range=0.2,                  # случайное приближение
    horizontal_flip=True              # отражение по горизонтали
)

# Для валидации – только нормализация (без аугментации)
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Загружаем изображения из папок
print("Загрузка тренировочных данных...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=True
)

print("Загрузка валидационных данных...")
validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

# ------------------------------
# 4. Создание модели (Transfer Learning)
# ------------------------------
# Загружаем предобученную MobileNetV2 без верхушки
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # замораживаем базовые слои

# Добавляем свои слои поверх
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Компиляция
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------
# 5. Обучение
# ------------------------------
print("Начало обучения...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
    ]
)

# ------------------------------
# 6. Сохранение модели
# ------------------------------
model_path = os.path.join(BASE_DIR, 'defect_model.h5')
model.save(model_path)
print(f"Модель сохранена как: {model_path}")

# ------------------------------
# 7. Оценка
# ------------------------------
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Итоговая точность на валидации: {val_acc:.4f}")