import asyncio
import logging
import os
import uuid
import cv2
import torch

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile

from ultralytics import YOLO

# --- Конфигурация ---
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TEMP_DIR = "temp_videos"

os.makedirs(TEMP_DIR, exist_ok=True)

# --- Инициализация ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# --- Загрузка модели YOLO ---
try:
    logging.info("Загрузка модели YOLO...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolo11n.pt').to(device)
    logging.info(f"Модель YOLO загружена и использует {device}.")
except Exception as e:
    logging.error(f"Ошибка загрузки модели YOLO: {e}")
    exit()

@dp.message(CommandStart())
async def send_welcome(message: types.Message):
    """Отправляет приветственное сообщение."""
    await message.reply("Привет! 👋\nОтправь мне видео (как файл или обычное видео), и я распознаю объекты с помощью YOLO.")

@dp.message(F.document | F.video)
async def handle_all_videos(message: types.Message):
    """Обрабатывает все типы видео (документ и обычное видео)"""
    user_id = message.from_user.id
    chat_id = message.chat.id
    message_id = message.message_id

    # Определение типа файла
    if message.video:
        file = message.video
        is_document = False
    else:
        file = message.document
        is_document = True

    # Проверка MIME-типа для документов
    if is_document and not (file.mime_type and file.mime_type.startswith('video/')):
        await message.reply("Пожалуйста, отправьте видеофайл.")
        return

    request_id = str(uuid.uuid4())
    input_video_path = os.path.join(TEMP_DIR, f"{request_id}_input.mp4")
    output_video_path = os.path.join(TEMP_DIR, f"{request_id}_output.mp4")

    try:
        await message.reply("Принял видео. Начинаю скачивание...")
        file_info = await bot.get_file(file.file_id)
        await bot.download_file(file_info.file_path, input_video_path)
        logging.info(f"Видео {file.file_name} ({file.file_size} байт) от {user_id} скачано в {input_video_path}")
        await message.reply("Видео скачано. Начинаю обработку с YOLO... ⏳")
    except Exception as e:
        logging.error(f"Ошибка скачивания видео от {user_id}: {e}")
        await message.reply("Не смог скачать видео. Попробуй еще раз.")
        return

    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError("Не удалось открыть видеофайл.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Параметры видео: {width}x{height}, {fps:.2f} FPS, {total_frames} кадров")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % int(fps) == 0:
                logging.info(f"Обработка кадра {frame_count}/{total_frames}")

            results = model(frame, device=device, verbose=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()
        logging.info(f"Обработка видео {request_id} завершена.")

    except Exception as e:
        logging.error(f"Ошибка обработки видео {request_id}: {e}")
        await message.reply("Ошибка обработки видео. 😢")
        if os.path.exists(input_video_path): os.remove(input_video_path)
        if os.path.exists(output_video_path): os.remove(output_video_path)
        return

    try:
        await message.reply("Обработка завершена! Отправляю результат...")
        output_file = FSInputFile(output_video_path, filename=f"processed_{file.file_name or 'video.mp4'}")
        await bot.send_video(chat_id=chat_id, video=output_file, caption="Готово! ✅", reply_to_message_id=message_id)
        logging.info(f"Видео {request_id} отправлено пользователю {user_id}")
    except Exception as e:
        logging.error(f"Ошибка отправки видео {request_id}: {e}")
        try:
            await message.reply("Пробую отправить как документ...")
            output_file_doc = FSInputFile(output_video_path, filename=f"processed_{file.file_name or 'video.mp4'}")
            await bot.send_document(chat_id=chat_id, document=output_file_doc, caption="Готово! ✅", reply_to_message_id=message_id)
        except Exception as e2:
            logging.error(f"Ошибка отправки документа {request_id}: {e2}")
            await message.reply("Не удалось отправить видео. Возможно, оно слишком большое.")
    finally:
        try:
            if os.path.exists(input_video_path): os.remove(input_video_path)
            if os.path.exists(output_video_path): os.remove(output_video_path)
            logging.info(f"Временные файлы {request_id} удалены.")
        except Exception as e:
            logging.error(f"Ошибка удаления файлов {request_id}: {e}")

@dp.message()
async def handle_other_messages(message: types.Message):
    """Обрабатывает некорректные сообщения."""
    await message.reply("Я обрабатываю только видео. Попробуй еще раз.")

# --- Запуск бота ---
async def main():
    logging.info("Запуск бота...")
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Бот остановлен.")
    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}")