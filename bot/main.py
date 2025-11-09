import os
import logging
import sys
import asyncio
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# Добавляем путь к корню проекта для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ml_worker.search import vectorize_face

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
TEMP_DIR = "data/photos/temporary"   # путь к временному хранилищу фоток, отправляемых пользователями

def get_main_keyboard():
    """Создает основную клавиатуру с двумя кнопками"""
    keyboard = [
        [KeyboardButton("Загрузить фото")],
        [KeyboardButton("Помощь")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = (
        "Привет!👋\n\n"
        "Я помогу найти ваши фото с мероприятия. Загрузите свое портретное фото — я найду все остальные."
    )

    reply_markup = get_main_keyboard()
    await update.message.reply_text(
        welcome_text,
        reply_markup=reply_markup
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик получения фото от пользователя"""
    user = update.effective_user
    logger.info(f"Пользователь {user.first_name} отправил фото")

    # Получаем файл фото
    photo = update.message.photo[-1]  # Берем фото наибольшего размера
    file = await context.bot.get_file(photo.file_id)

    # сохранить фото
    file_path = os.path.join(TEMP_DIR, f"{photo.file_id}.jpg")
    await file.download_to_drive(file_path)

    # Отправляем сообщение о поиске
    sent_message = await update.message.reply_text(
        "Подождите секундочку, ищу все ваши фотографии на посещенном мероприятии"
    )

    # Поиск фото по лицу
    search_result = vectorize_face(file_path)
    
    if search_result and search_result.get("user_folder"):
        # Используем найденную папку пользователя
        user_folder = search_result["user_folder"]
        await send_photos_from_folder(update, context, user_folder, sent_message)
    else:
        # Если поиск не дал результатов, сообщаем об этом
        error_msg = "К сожалению, не удалось найти ваши фотографии. Попробуйте загрузить другое фото."
        reply_markup = get_main_keyboard()
        if sent_message:
            await sent_message.edit_text(error_msg)
            await update.message.reply_text("Попробуйте снова.", reply_markup=reply_markup)
        else:
            await update.message.reply_text(error_msg, reply_markup=reply_markup)


async def send_photos_from_folder(update: Update, context: ContextTypes.DEFAULT_TYPE, folder_path: str,
                                  sent_message=None):
    """Отправляет несколько фото из указанной папки в виде файлов (для сохранения качества)"""
    try:
        # Проверяем, существует ли папка
        if not os.path.exists(folder_path):
            error_msg = (
                f"Ошибка: папка {folder_path} не найдена. "
            )
            reply_markup = get_main_keyboard()
            if sent_message:
                await sent_message.edit_text(error_msg)
                await update.message.reply_text("Попробуйте снова.", reply_markup=reply_markup)
            else:
                await update.message.reply_text(error_msg, reply_markup=reply_markup)
            return

        # Получаем список всех фото файлов
        photo_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        photo_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(photo_extensions) and os.path.isfile(os.path.join(folder_path, f))
        ]

        if not photo_files:
            error_msg = "В папке не найдено фото файлов."
            reply_markup = get_main_keyboard()
            if sent_message:
                await sent_message.edit_text(error_msg)
                await update.message.reply_text("Попробуйте снова.", reply_markup=reply_markup)
            else:
                await update.message.reply_text(error_msg, reply_markup=reply_markup)
            return

        # Удаляем сообщение "Подождите секундочку..."
        if sent_message:
            try:
                await sent_message.delete()
            except:
                pass

        # Функция для отправки одного фото
        async def send_single_photo(photo_file):
            photo_path = os.path.join(folder_path, photo_file)
            try:
                with open(photo_path, 'rb') as photo:
                    await update.message.reply_document(
                        document=photo,
                        filename=photo_file
                    )
                logger.info(f"Отправлено фото как файл: {photo_file}")
                return True
            except Exception as e:
                logger.error(f"Ошибка при отправке фото {photo_file}: {e}")
                return False

        # Отправляем все фото параллельно
        tasks = [send_single_photo(photo_file) for photo_file in photo_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_count = sum(1 for r in results if r is True)

        if successful_count > 0:
            await update.message.reply_text(f"Отправлено {successful_count} фото!")
            # Финальное сообщение от бота с кнопками
            final_keyboard = [
                [KeyboardButton("Да")],
                [
                    KeyboardButton("Нет, спасибо"),
                    KeyboardButton("Начать заново")
                ]
            ]
            final_reply_markup = ReplyKeyboardMarkup(final_keyboard, resize_keyboard=True, one_time_keyboard=False)
            await update.message.reply_text(
                "Сбой? Хотите повторить?",
                reply_markup=final_reply_markup
            )

    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}", exc_info=True)
        error_msg = "Произошла ошибка при отправке фото."
        reply_markup = get_main_keyboard()
        if sent_message:
            try:
                await sent_message.edit_text(error_msg)
                await update.message.reply_text("Попробуйте снова.", reply_markup=reply_markup)
            except:
                await update.message.reply_text(error_msg, reply_markup=reply_markup)
        else:
            await update.message.reply_text(error_msg, reply_markup=reply_markup)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    text = update.message.text.strip()

    # Обработка кнопки "Загрузить фото"
    if text == "Загрузить фото":
        reply_markup = get_main_keyboard()
        await update.message.reply_text(
            "Пожалуйста, отправьте мне ваше портретное фото.",
            reply_markup=reply_markup
        )
        return

    # Обработка кнопки "Помощь"
    if text == "Помощь":
        help_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Как это работает?", callback_data="help_how_it_works")],
            [InlineKeyboardButton("Вопросы?", callback_data="help_questions")]
        ])
        await update.message.reply_text(
            "Чем я могу помочь?",
            reply_markup=help_keyboard
        )
        return

    # Обработка кнопки "Да" (ответ на "Сбой? Хотите повторить?")
    if text == "Да" or text == "да":
        await start(update, context)
        return

    # Обработка кнопки "Начать заново"
    if text == "Начать заново":
        await start(update, context)
        return

    # Обработка кнопки "Нет, спасибо"
    if text == "Нет, спасибо":
        reply_markup = get_main_keyboard()
        await update.message.reply_text(
            "Спасибо, что использовали нас! Если понадоблюсь — я здесь.",
            reply_markup=reply_markup
        )
        return

    # Если пользователь написал текстовое сообщение (не кнопку)
    reply_markup = get_main_keyboard()
    await update.message.reply_text(
        "Пожалуйста, загрузите фото или используйте кнопки ниже.👇",
        reply_markup=reply_markup
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик callback-запросов от inline кнопок"""
    query = update.callback_query
    await query.answer()

    if query.data == "help_how_it_works":
        instructions = (
            "1️⃣ Загрузите ваше фото\n\n"
            "2️⃣ Я найду похожие снимки\n\n"
            "3️⃣ Скачайте все сразу"
        )
        await query.message.reply_text(instructions, reply_markup=get_main_keyboard())

    elif query.data == "help_questions":
        await query.message.reply_text(
            "Если у вас есть какой-то вопрос или предложение о сотрудничестве, напишите @imnomberone",
            reply_markup=get_main_keyboard()
        )


def main():
    """Запуск бота"""
    # Проверка токена
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or not BOT_TOKEN:
        print("\n" + "=" * 60)
        print("❌ ОШИБКА: Токен бота не установлен!")
        print("Установите токен в переменной BOT_TOKEN")
        print("=" * 60 + "\n")
        return

    # Создаем папку для фото если не существует
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)
        print(f"📁 Создана папка для временного хранения фото от пользователя: {TEMP_DIR}")

    # Создаем приложение
    application = Application.builder().token(BOT_TOKEN).build()

    # Регистрируем обработчики (важен порядок!)
    # -- обработчик команды /start
    application.add_handler(CommandHandler("start", start))
    # -- обработчик callback-запросов от inline кнопок
    application.add_handler(CallbackQueryHandler(handle_callback))
    # -- обработчик фото
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    # -- обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


    # Запускаем бота
    print("\n" + "=" * 60)
    print("✅ Бот запущен и готов к работе!")
    print("Нажмите Ctrl+C для остановки")
    print("=" * 60 + "\n")
    logger.info("Бот запущен...")

    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        print("\n\n⚠️  Бот остановлен пользователем")
        logger.info("Бот остановлен")


if __name__ == '__main__':
    main()

