# Telegram YOLO Video Processing Bot (v2)

Этот Telegram-бот принимает видеофайл или видео-сообщение, обрабатывает его с помощью модели обнаружения объектов YOLO, рисует ограничивающие рамки (bounding boxes) вокруг обнаруженных объектов и отправляет обработанное видео обратно пользователю.

Проект создан в ознакомительных целях для демонстрации интеграции YOLO с Telegram-ботом.

## 🤖 Что делает бот?

1.  **Принимает видео:** Пользователь отправляет боту видео **либо как файл (документ)**, **либо как обычное видео-сообщение** (например, записанное в Telegram или выбранное из галереи).
2.  **Скачивает видео:** Бот скачивает видео на сервер, где он запущен.
3.  **Обрабатывает кадры:** Бот читает видео покадрово. К каждому кадру применяется модель YOLO11n (`yolo11n.pt`) для обнаружения объектов (из стандартного набора COCO).
4.  **Рисует рамки:** На кадры наносятся прямоугольные рамки вокруг найденных объектов с подписями классов.
5.  **Собирает видео:** Обработанные кадры собираются в новый видеофайл.
6.  **Отправляет результат:** Готовое видео с разметкой отправляется обратно пользователю.

## ✨ Возможности

*   Обработка видео, отправленных **как документ (файл) или как обычное видео-сообщение**.
*   Использование предобученной модели YOLO11n (`yolo11n.pt`) из библиотеки `ultralytics`.
*   Автоматическое рисование рамок и меток на видео.
*   Работает асинхронно благодаря `aiogram`.
*   Очищает временные файлы после обработки.
*   Автоматическое определение устройства (CPU/GPU) для обработки.

## 🛠 Технологии

*   **Python 3.12+**
*   **Aiogram 3.x:** Современный асинхронный фреймворк для Telegram ботов.
*   **Ultralytics:** Библиотека для работы с моделями YOLO (включая YOLO11).
*   **OpenCV-Python (`cv2`):** Библиотека для компьютерного зрения, используется для чтения, обработки и записи видеокадров.
*   **PyTorch:** Основной фреймворк для `ultralytics` (устанавливается как зависимость).
*   **Aiofiles:** Для асинхронной работы с файлами (используется для скачивания).

## 📋 Предварительные требования

*   Установленный Python (версии 3.12 или новее рекомендуется).
*   `pip` (менеджер пакетов Python).
*   Токен для Telegram бота (получить у `@BotFather` в Telegram).

## 🚀 Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/sackvoich/YOLO_video_processing_bot
    cd YOLO_video_processing_bot
    ```

2.  **Создайте и активируйте виртуальное окружение (рекомендуется):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # MacOS/Linux
    source venv/bin/activate
    ```

3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```
    *Если файла `requirements.txt` нет, установите библиотеки вручную:*
    ```bash
    pip install aiogram ultralytics opencv-python aiofiles torch torchvision torchaudio
    ```
    *(Примечание: `torch`, `torchvision`, `torchaudio` являются зависимостями `ultralytics`. Иногда их лучше ставить отдельно согласно [официальной инструкции PyTorch](https://pytorch.org/get-started/locally/) для лучшей совместимости с вашим железом, особенно если планируете использовать GPU).*

## ⚙️ Конфигурация

1.  Откройте файл `bot.py` в текстовом редакторе.
2.  Найдите строку:
    ```python
    TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
    ```
3.  Замените `"YOUR_BOT_TOKEN_HERE"` на ваш реальный токен, полученный от `@BotFather`.
4.  (Опционально) Вы можете изменить папку для временных файлов, изменив значение переменной `TEMP_DIR`.

## ▶️ Запуск бота

Выполните в терминале (убедитесь, что виртуальное окружение активировано):

```bash
python bot.py
```

Бот запустится и начнет принимать обновления от Telegram. При первом запуске библиотека `ultralytics` может скачать файл весов модели `yolo11n.pt`, если он отсутствует.

## 💬 Как использовать

1.  Найдите вашего бота в Telegram по имени пользователя, которое вы задали в `@BotFather`.
2.  Начните диалог с ботом (можно отправить команду `/start`). Бот ответит:
    ```
    Привет! 👋
    Отправь мне видео (как файл или обычное видео), и я распознаю объекты с помощью YOLO.
    ```
3.  **Отправьте боту видео одним из способов:**
    *   **Как Файл (Документ):**
        *   Нажмите на значок "скрепки" (📎).
        *   Выберите "Файл" (File).
        *   Выберите нужное видео.
        *   **Важно:** Убедитесь, что опция "Сжать видео" (Compress video) *не* выбрана (если она появляется). Отправка как файл сохраняет лучшее качество.
    *   **Как Обычное Видео-сообщение:**
        *   Нажмите на значок "скрепки" (📎) и выберите видео из галереи *без* выбора опции "Файл".
        *   Или запишите видео прямо в чате Telegram и отправьте его.
4.  Бот напишет, что принял видео и начал обработку (`Принял видео. Начинаю скачивание...`, `Видео скачано. Начинаю обработку с YOLO... ⏳`).
5.  Дождитесь завершения. Время обработки зависит от длительности и разрешения видео, а также мощности вашего сервера (CPU/GPU).
6.  Бот пришлет обработанное видео (`Готово! ✅`) с размеченными объектами. Если отправка как видео не удалась (например, из-за размера), бот попытается отправить его как документ.

## ⚠️ Ограничения

*   **Время обработки:** Обработка видео - ресурсоемкая задача. На CPU она может быть очень медленной. Использование GPU (Nvidia с CUDA) значительно ускоряет процесс.
*   **Размер файла:** Telegram Bot API имеет ограничения на размер файлов (до 50 МБ для скачивания ботом и до 50 МБ / 2000 МБ (с Premium) для загрузки ботом). Бот может не справиться с очень большими видео. Также обработанное видео может быть больше исходного.
*   **Точность YOLO:** Используется модель `yolo11n` - самая быстрая, но наименее точная из семейства YOLO11. Для большей точности можно использовать другие модели (`yolo11s`, `yolo11m` и т.д.), но они требуют больше ресурсов и времени на обработку. Замените `'yolo11n.pt'` в коде на имя нужной модели (`model = YOLO('NAME_OF_YOUR_YOLO_MODEL').to(device)`).
*   **Параллельная обработка:** В текущей реализации обработка видео происходит последовательно внутри обработчика. Хотя `aiogram` асинхронен, тяжелая задача обработки одного видео может "заблокировать" обработку других запросов на некоторое время, пока она не завершится.
*   **Кодеки:** Используется кодек `mp4v` для записи. В редких случаях могут быть проблемы с воспроизведением итогового файла на некоторых устройствах.

## 💡 Возможные улучшения

*   Вынос тяжелой задачи обработки видео в отдельный асинхронный процесс или очередь задач (например, с использованием Celery, asyncio.to_thread, или `multiprocessing`), чтобы не блокировать основной цикл бота.
*   Добавление выбора модели YOLO пользователем через команды или кнопки.
*   Настройка параметров обработки (например, порог уверенности (`conf`) для детекций) через команды.
*   Более гибкая конфигурация (через переменные окружения или `.env` файл).
*   Развертывание с использованием Docker.
*   Улучшенная обработка ошибок и более подробное информирование пользователя о статусе (например, % выполнения).
*   Добавление обработки "кружков" (video notes), если это необходимо (требует отдельного хендлера и, возможно, конвертации).

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. Подробности см. в файле `LICENSE` (если он есть) или ознакомьтесь с текстом лицензии MIT.
