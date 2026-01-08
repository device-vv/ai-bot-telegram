import asyncio
import json
import sqlite3
import logging
import io
import re
import base64
from datetime import datetime
from typing import Optional, List, Dict
import aiohttp
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ADMIN_PASSWORD = "Nova306"
BOT_NAME = "Nova"
DEFAULT_GROUP_ID = -1004869718058
MAX_MESSAGE_LENGTH = 4096
MAX_FILE_SIZE = 20 * 1024 * 1024

class ConfigStates(StatesGroup):
    waiting_for_api_key = State()
    waiting_for_model = State()
    waiting_for_password = State()

def markdown_to_telegram(text: str) -> str:
    """Конвертирует обычный Markdown в Telegram MarkdownV2"""

    # ИСПРАВЛЕНО: Используем маркер без подчеркиваний
    # Сохраняем блоки кода (их не трогаем)
    code_blocks = []
    def save_code(match):
        code_blocks.append(match.group(0))
        return f"§§§CODEBLOCK{len(code_blocks)-1}§§§"

    text = re.sub(r'```[\w]*\n.*?```', save_code, text, flags=re.DOTALL)

    # Сохраняем инлайн код
    inline_codes = []
    def save_inline(match):
        inline_codes.append(match.group(0))
        return f"§§§INLINECODE{len(inline_codes)-1}§§§"

    text = re.sub(r'`[^`]+`', save_inline, text)

    # Преобразуем Markdown заголовки в жирный текст для Telegram
    text = re.sub(r'^####\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)

    # Преобразуем **жирный** в *жирный* для Telegram MarkdownV2
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    text = re.sub(r'__(.+?)__', r'*\1*', text)

    # Курсив: *курсив* -> _курсив_ для Telegram
    text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'_\1_', text)

    # Экранируем спецсимволы, НЕ внутри тегов форматирования
    def escape_text(txt):
        chars_to_escape = r'_*[]()~`>#+-=|{}.!\\'
        result = ''
        i = 0
        while i < len(txt):
            # ИСПРАВЛЕНО: Проверяем плейсхолдеры ПЕРВЫМ ДЕЛОМ
            if txt[i:i+3] == '§§§':
                # Ищем закрывающий §§§
                end = txt.find('§§§', i+3)
                if end != -1:
                    result += txt[i:end+3]
                    i = end + 3
                    continue

            char = txt[i]

            # Проверяем теги форматирования
            if char == '*':
                end = txt.find('*', i+1)
                if end != -1:
                    content = txt[i+1:end]
                    escaped_content = ''
                    for c in content:
                        if c in '[]()~`>#+-=|{}.!\\':
                            escaped_content += '\\' + c
                        else:
                            escaped_content += c
                    result += '*' + escaped_content + '*'
                    i = end + 1
                    continue
            elif char == '_':
                end = txt.find('_', i+1)
                if end != -1:
                    content = txt[i+1:end]
                    escaped_content = ''
                    for c in content:
                        if c in '[]()~`>#+-=|{}.!\\':
                            escaped_content += '\\' + c
                        else:
                            escaped_content += c
                    result += '_' + escaped_content + '_'
                    i = end + 1
                    continue

            # Обычный символ - экранируем если нужно
            if char in chars_to_escape:
                result += '\\' + char
            else:
                result += char
            i += 1

        return result

    lines = text.split('\n')
    processed_lines = []

    for line in lines:
        # Если это плейсхолдер кода, не трогаем
        if '§§§CODEBLOCK' in line or '§§§INLINECODE' in line:
            processed_lines.append(line)
        else:
            processed_lines.append(escape_text(line))

    text = '\n'.join(processed_lines)

    # Возвращаем блоки кода обратно
    for i, code in enumerate(code_blocks):
        text = text.replace(f"§§§CODEBLOCK{i}§§§", code)

    # Возвращаем инлайн код
    for i, code in enumerate(inline_codes):
        text = text.replace(f"§§§INLINECODE{i}§§§", code)

    return text

async def send_markdown_message(message: Message, text: str, reply: bool = False):
    """Отправка сообщения в MarkdownV2"""

    # Конвертируем Markdown в Telegram MarkdownV2
    try:
        formatted_text = markdown_to_telegram(text)
    except Exception as e:
        logger.error(f"Error converting markdown: {e}")
        formatted_text = text

    # Разбиваем на части если нужно
    parts = []
    max_len = MAX_MESSAGE_LENGTH

    if len(formatted_text) <= max_len:
        parts = [formatted_text]
    else:
        lines = formatted_text.split('\n')
        current = ''
        for line in lines:
            if len(current) + len(line) + 1 > max_len:
                if current:
                    parts.append(current)
                current = line
            else:
                current = (current + '\n' if current else '') + line
        if current:
            parts.append(current)

    logger.info(f"Sending message in {len(parts)} part(s)")

    for i, part in enumerate(parts):
        try:
            if i == 0 and reply:
                await message.reply(part, parse_mode=ParseMode.MARKDOWN_V2)
            else:
                await message.answer(part, parse_mode=ParseMode.MARKDOWN_V2)
            logger.info(f"Sent part {i+1}/{len(parts)}")

            if i < len(parts) - 1:
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error sending MarkdownV2 part {i+1}: {e}")
            try:
                # Fallback: отправляем без форматирования
                plain_text = re.sub(r'[*_`\[\]()~>#+=|{}.!\\-]', '', part)
                if i == 0 and reply:
                    await message.reply(plain_text, parse_mode=None)
                else:
                    await message.answer(plain_text, parse_mode=None)
                logger.info(f"Sent part {i+1} without formatting")
            except Exception as ex:
                logger.error(f"Failed to send: {ex}")

class Database:
    def __init__(self, db_name: str = "nova_bot.db"):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY, api_key TEXT NOT NULL,
            model TEXT DEFAULT 'gpt-4.1-nano',
            bot_enabled INTEGER DEFAULT 1,
            base_url TEXT DEFAULT 'https://api.proxyapi.ru/openai/v1')''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS admins (
            user_id INTEGER PRIMARY KEY, username TEXT,
            first_name TEXT, last_name TEXT,
            logged_in_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS allowed_groups (
            group_id INTEGER PRIMARY KEY, group_name TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER,
            user_id INTEGER, username TEXT, message TEXT,
            response TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        cursor.execute('INSERT OR IGNORE INTO allowed_groups (group_id, group_name) VALUES (?, ?)', (DEFAULT_GROUP_ID, 'Default'))
        cursor.execute('SELECT COUNT(*) FROM settings')
        if cursor.fetchone()[0] == 0:
            cursor.execute('INSERT INTO settings (id, api_key) VALUES (1, ?)', ('YOUR_API_KEY_HERE',))

        conn.commit()
        conn.close()

    def get_settings(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT api_key, model, bot_enabled, base_url FROM settings WHERE id=1')
        r = c.fetchone()
        conn.close()
        return {'api_key': r[0], 'model': r[1], 'bot_enabled': bool(r[2]), 'base_url': r[3]} if r else None

    def update_api_key(self, api_key):
        conn = sqlite3.connect(self.db_name)
        conn.execute('UPDATE settings SET api_key=? WHERE id=1', (api_key,))
        conn.commit()
        conn.close()

    def update_model(self, model):
        conn = sqlite3.connect(self.db_name)
        conn.execute('UPDATE settings SET model=? WHERE id=1', (model,))
        conn.commit()
        conn.close()

    def toggle_bot(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT bot_enabled FROM settings WHERE id=1')
        cur = c.fetchone()[0]
        new = 0 if cur else 1
        conn.execute('UPDATE settings SET bot_enabled=? WHERE id=1', (new,))
        conn.commit()
        conn.close()
        return bool(new)

    def is_admin(self, uid):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM admins WHERE user_id=?', (uid,))
        r = c.fetchone()[0] > 0
        conn.close()
        return r

    def add_admin(self, uid, username=None, fname=None, lname=None):
        conn = sqlite3.connect(self.db_name)
        conn.execute('INSERT OR REPLACE INTO admins VALUES (?,?,?,?,CURRENT_TIMESTAMP)', (uid, username, fname, lname))
        conn.commit()
        conn.close()

    def remove_admin(self, uid):
        conn = sqlite3.connect(self.db_name)
        conn.execute('DELETE FROM admins WHERE user_id=?', (uid,))
        conn.commit()
        conn.close()

    def get_admins(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT * FROM admins')
        r = c.fetchall()
        conn.close()
        return [{'user_id': x[0], 'username': x[1], 'first_name': x[2], 'last_name': x[3], 'logged_in_at': x[4]} for x in r]

    def is_group_allowed(self, gid):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM allowed_groups WHERE group_id=?', (gid,))
        r = c.fetchone()[0] > 0
        conn.close()
        return r

    def add_group(self, gid, gname=None):
        conn = sqlite3.connect(self.db_name)
        conn.execute('INSERT OR IGNORE INTO allowed_groups (group_id, group_name) VALUES (?,?)', (gid, gname))
        conn.commit()
        conn.close()

    def remove_group(self, gid):
        conn = sqlite3.connect(self.db_name)
        conn.execute('DELETE FROM allowed_groups WHERE group_id=?', (gid,))
        conn.commit()
        conn.close()

    def get_allowed_groups(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT * FROM allowed_groups')
        r = c.fetchall()
        conn.close()
        return [{'group_id': x[0], 'group_name': x[1], 'added_at': x[2]} for x in r]

    def save_chat_history(self, cid, uid, uname, msg, resp):
        conn = sqlite3.connect(self.db_name)
        conn.execute('INSERT INTO chat_history (chat_id, user_id, username, message, response) VALUES (?,?,?,?,?)', 
                     (cid, uid, uname, msg, resp))
        conn.commit()
        conn.close()

    def get_chat_history(self, cid, limit=10):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT message, response FROM chat_history WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?', (cid, limit))
        r = c.fetchall()
        conn.close()
        return [{'message': x[0], 'response': x[1]} for x in reversed(r)]

class NovaAI:
    def __init__(self, db):
        self.db = db

    async def analyze_image(self, img_data, caption=None):
        s = self.db.get_settings()
        if not s['api_key'] or s['api_key'] == 'YOUR_API_KEY_HERE':
            return None

        try:
            b64 = base64.b64encode(img_data).decode('utf-8')
            prompt = caption + "\n\nОпиши подробно картинку." if caption else "Опиши подробно картинку."

            async with aiohttp.ClientSession() as session:
                data = {
                    "model": "gpt-4o-mini",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]
                    }],
                    "max_tokens": 1000
                }

                async with session.post(f"{s['base_url']}/chat/completions",
                                      headers={"Authorization": f"Bearer {s['api_key']}", "Content-Type": "application/json"},
                                      json=data, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        r = await resp.json()
                        return r['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
        return None

    async def transcribe_audio(self, audio_data, fname="audio.ogg"):
        s = self.db.get_settings()
        if not s['api_key'] or s['api_key'] == 'YOUR_API_KEY_HERE':
            return None

        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field('file', audio_data, filename=fname, content_type='audio/ogg')
                data.add_field('model', 'whisper-1')
                data.add_field('language', 'ru')

                async with session.post(f"{s['base_url']}/audio/transcriptions",
                                      headers={"Authorization": f"Bearer {s['api_key']}"},
                                      data=data, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        r = await resp.json()
                        return r.get('text', '')
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        return None

    async def analyze_document(self, content, fname):
        prompt = f"Проанализируй файл '{fname}':\n\n{content[:3000]}..."
        return await self.generate_response(prompt, 0)

    async def generate_response(self, msg, cid):
        s = self.db.get_settings()
        if not s['api_key'] or s['api_key'] == 'YOUR_API_KEY_HERE':
            return "⚠️ API ключ не настроен. /setapi"

        hist = self.db.get_chat_history(cid, limit=5)

        messages = [{
            "role": "system",
            "content": f"Ты — {BOT_NAME}, дружелюбный AI-ассистент. Отвечай на русском. Используй Markdown: **жирный**, *курсив*, `код`, ```блоки кода``` с языком (```python, ```html). Используй заголовки ## и ###. Структурируй ответы."
        }]

        for h in hist:
            messages.append({"role": "user", "content": h['message']})
            messages.append({"role": "assistant", "content": h['response']})

        messages.append({"role": "user", "content": msg})

        try:
            async with aiohttp.ClientSession() as session:
                data = {"model": s['model'], "messages": messages, "temperature": 0.7, "max_tokens": 2000}

                async with session.post(f"{s['base_url']}/chat/completions",
                                      headers={"Authorization": f"Bearer {s['api_key']}", "Content-Type": "application/json"},
                                      json=data, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        r = await resp.json()
                        return r['choices'][0]['message']['content']
                    else:
                        return f"❌ API error: {resp.status}"
        except asyncio.TimeoutError:
            return "⏱️ Timeout"
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"❌ Error: {str(e)}"

class NovaBot:
    def __init__(self, token):
        self.bot = Bot(token=token, default=DefaultBotProperties())
        self.dp = Dispatcher(storage=MemoryStorage())
        self.db = Database()
        self.ai = NovaAI(self.db)
        self.setup_handlers()

    def setup_handlers(self):
        self.dp.message.register(self.cmd_start, Command("start"))
        self.dp.message.register(self.cmd_login, Command("login"))
        self.dp.message.register(self.cmd_logout, Command("logout"))
        self.dp.message.register(self.cmd_admin, Command("admin"))
        self.dp.message.register(self.cmd_setapi, Command("setapi"))
        self.dp.message.register(self.cmd_setmodel, Command("setmodel"))
        self.dp.message.register(self.cmd_toggle, Command("toggle"))
        self.dp.message.register(self.cmd_status, Command("status"))
        self.dp.message.register(self.cmd_addgroup, Command("addgroup"))
        self.dp.message.register(self.cmd_removegroup, Command("removegroup"))
        self.dp.message.register(self.cmd_groups, Command("groups"))
        self.dp.message.register(self.cmd_admins, Command("admins"))
        self.dp.message.register(self.handle_photo, F.photo)
        self.dp.message.register(self.handle_voice, F.voice)
        self.dp.message.register(self.handle_document, F.document)
        self.dp.message.register(self.handle_group_message, F.chat.type.in_({"group", "supergroup"}))
        self.dp.message.register(self.handle_private_message, F.chat.type == "private")
        self.dp.callback_query.register(self.handle_callback)

    async def handle_photo(self, msg):
        if not self.db.get_settings()['bot_enabled']:
            return
        if msg.chat.type in ["group", "supergroup"] and not self.db.is_group_allowed(msg.chat.id):
            return

        try:
            await msg.answer("🖼️ Анализирую...")
            photo = msg.photo[-1]
            pf = await self.bot.get_file(photo.file_id)
            pb = io.BytesIO()
            await self.bot.download_file(pf.file_path, pb)

            analysis = await self.ai.analyze_image(pb.getvalue(), msg.caption)

            if analysis:
                self.db.save_chat_history(msg.chat.id, msg.from_user.id,
                                        msg.from_user.username or msg.from_user.first_name,
                                        f"[Фото]: {msg.caption or 'нет'}", analysis)

                await send_markdown_message(msg, f"🖼️ **Анализ изображения:**\n\n{analysis}", reply=True)
            else:
                await msg.reply("❌ Ошибка анализа")
        except Exception as e:
            logger.error(f"Photo error: {e}")
            await msg.reply(f"❌ Error: {e}")

    async def handle_voice(self, msg):
        if not self.db.get_settings()['bot_enabled']:
            return
        if msg.chat.type in ["group", "supergroup"] and not self.db.is_group_allowed(msg.chat.id):
            return

        try:
            await msg.answer("🎤 Распознаю...")
            vf = await self.bot.get_file(msg.voice.file_id)
            vb = io.BytesIO()
            await self.bot.download_file(vf.file_path, vb)

            trans = await self.ai.transcribe_audio(vb.getvalue())

            if trans:
                resp = await self.ai.generate_response(f"Голос: {trans}", msg.chat.id)
                self.db.save_chat_history(msg.chat.id, msg.from_user.id,
                                        msg.from_user.username or msg.from_user.first_name,
                                        f"[Голос]: {trans}", resp)

                await send_markdown_message(msg, f"📝 *Вы сказали:* {trans}\n\n{resp}", reply=True)
            else:
                await msg.reply("❌ Ошибка")
        except Exception as e:
            logger.error(f"Voice error: {e}")
            await msg.reply(f"❌ Error: {e}")

    async def handle_document(self, msg):
        if not self.db.get_settings()['bot_enabled']:
            return
        if msg.chat.type in ["group", "supergroup"] and not self.db.is_group_allowed(msg.chat.id):
            return

        try:
            doc = msg.document
            if doc.file_size > MAX_FILE_SIZE:
                await msg.reply(f"❌ Файл > {MAX_FILE_SIZE//(1024*1024)} MB")
                return

            exts = ['.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.md', '.csv']
            if not any(doc.file_name.endswith(e) for e in exts):
                await msg.reply(f"❌ Только: {', '.join(exts)}")
                return

            await msg.answer(f"📄 Анализирую {doc.file_name}...")

            f = await self.bot.get_file(doc.file_id)
            fb = io.BytesIO()
            await self.bot.download_file(f.file_path, fb)

            try:
                content = fb.getvalue().decode('utf-8')
            except:
                content = fb.getvalue().decode('latin-1')

            resp = await self.ai.analyze_document(content, doc.file_name)
            self.db.save_chat_history(msg.chat.id, msg.from_user.id,
                                    msg.from_user.username or msg.from_user.first_name,
                                    f"[Файл]: {doc.file_name}", resp)

            await send_markdown_message(msg, f"📁 **Анализ файла:** {doc.file_name}\n\n{resp}", reply=True)
        except Exception as e:
            logger.error(f"Doc error: {e}")
            await msg.reply(f"❌ Error: {e}")

    async def cmd_start(self, msg):
        admin_txt = "\n\n🔐 Вы админ. /admin" if self.db.is_admin(msg.from_user.id) else "\n\n🔐 Доступ: /login"
        text = (f"👋 Привет! Я {BOT_NAME} — AI-ассистент.\n\n"
                f"📝 В группе: упомяни '{BOT_NAME}'\n"
                f"💬 Личка: пиши напрямую\n"
                f"🎤 Голосовые сообщения\n"
                f"📄 Файлы (.txt, .py и т.д.)\n"
                f"🖼️ Фотографии{admin_txt}")
        await send_markdown_message(msg, text)

    async def cmd_login(self, msg, state):
        if self.db.is_admin(msg.from_user.id):
            await msg.answer("✅ Вы авторизованы")
            return

        args = msg.text.split(maxsplit=1)
        if len(args) > 1:
            if args[1].strip() == ADMIN_PASSWORD:
                self.db.add_admin(msg.from_user.id, msg.from_user.username, msg.from_user.first_name, msg.from_user.last_name)
                await msg.answer(f"✅ Успешно! /admin")
                try:
                    await msg.delete()
                except:
                    pass
            else:
                await msg.answer("❌ Неверный пароль")
        else:
            await msg.answer("🔐 Формат: /login ПАРОЛЬ")
            await state.set_state(ConfigStates.waiting_for_password)

    async def cmd_logout(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        self.db.remove_admin(msg.from_user.id)
        await msg.answer("👋 Вышли")

    async def cmd_admin(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            await msg.answer("⛔ Только админ")
            return

        s = self.db.get_settings()
        status = "🟢 Вкл" if s['bot_enabled'] else "🔴 Выкл"

        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Вкл/Выкл", callback_data="toggle_bot")],
            [InlineKeyboardButton(text="🔑 API", callback_data="change_api")],
            [InlineKeyboardButton(text="🤖 Модель", callback_data="change_model")],
            [InlineKeyboardButton(text="📊 Статус", callback_data="show_status")],
            [InlineKeyboardButton(text="👥 Группы", callback_data="manage_groups")],
            [InlineKeyboardButton(text="🔐 Админы", callback_data="show_admins")]
        ])

        text = f"⚙️ **Панель {BOT_NAME}**\n\nСтатус: {status}\nМодель: {s['model']}\nAPI: {'✅' if s['api_key'] != 'YOUR_API_KEY_HERE' else '❌'}"
        await send_markdown_message(msg, text)
        await msg.answer("Выберите действие:", reply_markup=kb)

    async def cmd_setapi(self, msg, state):
        if not self.db.is_admin(msg.from_user.id):
            return

        args = msg.text.split(maxsplit=1)
        if len(args) > 1:
            self.db.update_api_key(args[1].strip())
            await msg.answer("✅ API обновлен")
            try:
                await msg.delete()
            except:
                pass
        else:
            await msg.answer("🔑 Отправьте API ключ:")
            await state.set_state(ConfigStates.waiting_for_api_key)

    async def cmd_setmodel(self, msg, state):
        if not self.db.is_admin(msg.from_user.id):
            return

        args = msg.text.split(maxsplit=1)
        if len(args) > 1:
            self.db.update_model(args[1].strip())
            await msg.answer(f"✅ Модель: {args[1].strip()}")
        else:
            await msg.answer("🤖 Отправьте модель (например: gpt-4.1-nano)")
            await state.set_state(ConfigStates.waiting_for_model)

    async def cmd_toggle(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        new = self.db.toggle_bot()
        await msg.answer(f"Бот {'🟢 Вкл' if new else '🔴 Выкл'}")

    async def cmd_status(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        s = self.db.get_settings()
        g = self.db.get_allowed_groups()
        a = self.db.get_admins()
        text = f"📊 **Статус {BOT_NAME}**\n\nСостояние: {'🟢 Вкл' if s['bot_enabled'] else '🔴 Выкл'}\nМодель: {s['model']}\nГрупп: {len(g)}\nАдминов: {len(a)}"
        await send_markdown_message(msg, text)

    async def cmd_addgroup(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        if msg.chat.type in ["group", "supergroup"]:
            self.db.add_group(msg.chat.id, msg.chat.title)
            await msg.answer(f"✅ Группа '{msg.chat.title}' добавлена")
        else:
            args = msg.text.split(maxsplit=1)
            if len(args) > 1:
                try:
                    gid = int(args[1])
                    self.db.add_group(gid)
                    await msg.answer(f"✅ Группа {gid} добавлена")
                except:
                    await msg.answer("❌ Неверный ID")
            else:
                await msg.answer("📝 Используйте в группе или: /addgroup -100...")

    async def cmd_removegroup(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        if msg.chat.type in ["group", "supergroup"]:
            self.db.remove_group(msg.chat.id)
            await msg.answer(f"✅ Группа '{msg.chat.title}' удалена")
        else:
            args = msg.text.split(maxsplit=1)
            if len(args) > 1:
                try:
                    gid = int(args[1])
                    self.db.remove_group(gid)
                    await msg.answer(f"✅ Группа {gid} удалена")
                except:
                    await msg.answer("❌ Неверный ID")
            else:
                await msg.answer("📝 Используйте в группе или: /removegroup -100...")

    async def cmd_groups(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        g = self.db.get_allowed_groups()
        if not g:
            await msg.answer("📝 Нет групп")
            return
        text = "**Разрешенные группы:**\n\n"
        for x in g:
            text += f"• {x['group_name'] or 'Без названия'}\n  ID: {x['group_id']}\n"
        await send_markdown_message(msg, text)

    async def cmd_admins(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        a = self.db.get_admins()
        if not a:
            await msg.answer("📝 Нет админов")
            return
        text = "**Администраторы:**\n\n"
        for x in a:
            name = x['first_name'] or 'Без имени'
            uname = f"@{x['username']}" if x['username'] else 'без username'
            text += f"• {name} ({uname})\n  ID: {x['user_id']}\n"
        await send_markdown_message(msg, text)

    async def handle_group_message(self, msg):
        s = self.db.get_settings()
        if not s['bot_enabled'] or not self.db.is_group_allowed(msg.chat.id):
            return
        if not msg.text or BOT_NAME.lower() not in msg.text.lower():
            return

        umsg = msg.text
        for v in [BOT_NAME, BOT_NAME.lower(), BOT_NAME.upper()]:
            umsg = umsg.replace(v, "").strip()
        if not umsg:
            umsg = "Привет!"

        await msg.chat.do("typing")
        resp = await self.ai.generate_response(umsg, msg.chat.id)
        self.db.save_chat_history(msg.chat.id, msg.from_user.id,
                                 msg.from_user.username or msg.from_user.first_name, umsg, resp)
        await send_markdown_message(msg, resp, reply=True)

    async def handle_private_message(self, msg):
        s = self.db.get_settings()
        if not s['bot_enabled']:
            await msg.answer("🔴 Бот выключен")
            return
        if not msg.text:
            return

        await msg.chat.do("typing")
        resp = await self.ai.generate_response(msg.text, msg.chat.id)
        self.db.save_chat_history(msg.chat.id, msg.from_user.id,
                                 msg.from_user.username or msg.from_user.first_name, msg.text, resp)
        await send_markdown_message(msg, resp)

    async def handle_callback(self, cb, state):
        if not self.db.is_admin(cb.from_user.id):
            await cb.answer("⛔ Только админ", show_alert=True)
            return

        if cb.data == "toggle_bot":
            new = self.db.toggle_bot()
            await cb.answer(f"Бот {'🟢' if new else '🔴'}")
        elif cb.data == "change_api":
            await cb.message.answer("🔑 Отправьте API ключ:")
            await state.set_state(ConfigStates.waiting_for_api_key)
            await cb.answer()
        elif cb.data == "change_model":
            await cb.message.answer("🤖 Отправьте модель:")
            await state.set_state(ConfigStates.waiting_for_model)
            await cb.answer()
        elif cb.data == "show_status":
            s = self.db.get_settings()
            g = self.db.get_allowed_groups()
            a = self.db.get_admins()
            text = f"📊 **Статус**\n\nСостояние: {'🟢' if s['bot_enabled'] else '🔴'}\nМодель: {s['model']}\nГрупп: {len(g)}\nАдминов: {len(a)}"
            await send_markdown_message(cb.message, text)
            await cb.answer()
        elif cb.data == "manage_groups":
            g = self.db.get_allowed_groups()
            if not g:
                await cb.message.answer("📝 Нет групп")
            else:
                text = "**Группы:**\n\n"
                for x in g:
                    text += f"• {x['group_name'] or 'Без названия'}\n"
                await send_markdown_message(cb.message, text)
            await cb.answer()
        elif cb.data == "show_admins":
            a = self.db.get_admins()
            if not a:
                await cb.message.answer("📝 Нет админов")
            else:
                text = "**Админы:**\n\n"
                for x in a:
                    name = x['first_name'] or 'Без имени'
                    text += f"• {name}\n"
                await send_markdown_message(cb.message, text)
            await cb.answer()

    async def start(self):
        @self.dp.message(ConfigStates.waiting_for_password)
        async def proc_pass(msg, state):
            if msg.text.strip() == ADMIN_PASSWORD:
                self.db.add_admin(msg.from_user.id, msg.from_user.username, msg.from_user.first_name, msg.from_user.last_name)
                await msg.answer("✅ Успешно! /admin")
                await state.clear()
            else:
                await msg.answer("❌ Неверный пароль")
                await state.clear()
            try:
                await msg.delete()
            except:
                pass

        @self.dp.message(ConfigStates.waiting_for_api_key)
        async def proc_api(msg, state):
            if self.db.is_admin(msg.from_user.id):
                self.db.update_api_key(msg.text.strip())
                await msg.answer("✅ API обновлен")
                await state.clear()
                try:
                    await msg.delete()
                except:
                    pass

        @self.dp.message(ConfigStates.waiting_for_model)
        async def proc_model(msg, state):
            if self.db.is_admin(msg.from_user.id):
                self.db.update_model(msg.text.strip())
                await msg.answer(f"✅ Модель: {msg.text.strip()}")
                await state.clear()

        logger.info(f"Starting {BOT_NAME}...")
        await self.dp.start_polling(self.bot)

async def main():
    TOKEN = "8469411816:AAGOqJezsce0jgi1ZefbFKLM1oFdhF_JIGw"
    bot = NovaBot(TOKEN)
    await bot.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped")

