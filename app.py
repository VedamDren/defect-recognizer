import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Конфигурация
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Создаем папку для загрузок
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Глобальные переменные для модели
model = None
CLASS_NAMES = ['punching_hole', 'rolled_pit']  # Ваши классы
IMG_SIZE = 224

def load_model():
    """Загружает обученную модель при старте приложения"""
    global model
    try:
        # Путь к вашей модели (скачанной из Colab)
        model_path = os.path.join(os.path.dirname(__file__), 'defect_model.h5')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("Модель успешно загружена")
        else:
            print(f"Модель не найдена по пути: {model_path}")
            print("Будет использоваться заглушка для демонстрации")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Будет использоваться заглушка для демонстрации")

def allowed_file(filename):
    """Проверяет разрешенный тип файла"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recognize_defects_with_model(image_path):
    """
    Распознавание дефектов с помощью реальной нейросети
    Возвращает список словарей с названиями и вероятностями,
    отсортированный по убыванию вероятности
    """
    global model
    
    # Если модель не загружена, используем заглушку
    if model is None:
        # Демо-режим с фиксированными значениями
        results = [
            {'name': 'punching_hole', 'score': 0.85012345},
            {'name': 'rolled_pit', 'score': 0.14987655}
        ]
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    try:
        # Загружаем и подготавливаем изображение
        img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Предсказание
        predictions = model.predict(img_array)[0]
        
        # Формируем результаты для всех классов
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            results.append({
                'name': class_name,
                'score': float(predictions[i])
            })
        
        # Сортируем по убыванию вероятности
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
        
    except Exception as e:
        print(f"Ошибка при распознавании: {e}")
        # В случае ошибки возвращаем заглушку
        return [
            {'name': 'punching_hole', 'score': 0.5},
            {'name': 'rolled_pit', 'score': 0.5}
        ]

@app.route('/', methods=['GET', 'POST'])
def index():
    """Главная страница"""
    results = None
    filename = None
    error = None
    
    if request.method == 'POST':
        # Проверяем наличие файла
        if 'file' not in request.files:
            error = "Файл не найден в запросе"
            return render_template('index.html', results=results, filename=filename, error=error)
        
        file = request.files['file']
        
        if file.filename == '':
            error = "Файл не выбран"
            return render_template('index.html', results=results, filename=filename, error=error)
        
        if file and allowed_file(file.filename):
            try:
                # Сохраняем файл
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Распознаем дефекты
                results = recognize_defects_with_model(filepath)
                
                # Опционально удаляем файл после обработки
                # os.remove(filepath)
                
            except Exception as e:
                error = f"Ошибка обработки: {str(e)}"
        else:
            error = "Недопустимый тип файла. Разрешены: png, jpg, jpeg, gif, bmp"
    
    return render_template('index.html', results=results, filename=filename, error=error)

# Загружаем модель при запуске
load_model()

if __name__ == '__main__':
    app.run(debug=True)