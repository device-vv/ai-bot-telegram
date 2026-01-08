import asyncio
import json
import os
import sqlite3
import logging
import io
import re
import base64
from datetime import datetime
from typing import Optional, List, Dict
import aiohttp
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, BotCommand
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change_me_immediately")
BOT_NAME = "Nova"
DEFAULT_GROUP_ID = int(os.getenv("DEFAULT_GROUP_ID", "0"))
MAX_MESSAGE_LENGTH = 4096
MAX_FILE_SIZE = 20 * 1024 * 1024

class ConfigStates(StatesGroup):
    waiting_for_api_key = State()
    waiting_for_model = State()
    waiting_for_password = State()
    waiting_for_api_url = State()
    waiting_for_broadcast_message = State()
    waiting_for_student_name = State()
    waiting_for_homework_title = State()

def markdown_to_html(text: str) -> str:
    """Convert markdown to Telegram HTML format."""
    import html
    
    # Extract code blocks first to protect them
    code_blocks = []
    def save_code_block(match):
        lang = match.group(1) or ''
        code = match.group(2)
        # Escape HTML inside code
        escaped_code = html.escape(code)
        code_blocks.append(f'<pre><code class="language-{lang}">{escaped_code}</code></pre>' if lang else f'<pre>{escaped_code}</pre>')
        return f'\x00CODE{len(code_blocks)-1}\x00'
    
    text = re.sub(r'```(\w*)\n?([\s\S]*?)```', save_code_block, text)
    
    # Extract inline code
    inline_codes = []
    def save_inline(match):
        code = html.escape(match.group(1))
        inline_codes.append(f'<code>{code}</code>')
        return f'\x00INLINE{len(inline_codes)-1}\x00'
    
    text = re.sub(r'`([^`\n]+)`', save_inline, text)
    
    # Escape HTML in remaining text (but preserve placeholders)
    lines = text.split('\n')
    result_lines = []
    for line in lines:
        if '\x00' in line:
            # Line has placeholder - escape parts around it
            parts = re.split(r'(\x00\w+\d+\x00)', line)
            escaped_parts = []
            for part in parts:
                if part.startswith('\x00') and part.endswith('\x00'):
                    escaped_parts.append(part)
                else:
                    escaped_parts.append(html.escape(part))
            result_lines.append(''.join(escaped_parts))
        else:
            result_lines.append(html.escape(line))
    text = '\n'.join(result_lines)
    
    # Convert markdown formatting to HTML
    # Headers to bold
    text = re.sub(r'^#{1,4}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)
    
    # Restore code blocks
    for i, code in enumerate(code_blocks):
        text = text.replace(f'\x00CODE{i}\x00', code)
    
    # Restore inline code
    for i, code in enumerate(inline_codes):
        text = text.replace(f'\x00INLINE{i}\x00', code)
    
    return text

async def send_markdown_message(message: Message, text: str, reply: bool = False):
    """Send message with HTML formatting."""
    try:
        formatted_text = markdown_to_html(text)
    except Exception as e:
        logger.error(f"Error converting to HTML: {e}")
        formatted_text = text

    parts = []
    max_len = MAX_MESSAGE_LENGTH

    if len(formatted_text) <= max_len:
        parts = [formatted_text]
    else:
        # Split by lines, keeping code blocks together
        lines = formatted_text.split('\n')
        current = ''
        in_code_block = False
        for line in lines:
            if '<pre>' in line:
                in_code_block = True
            if '</pre>' in line:
                in_code_block = False
            
            if len(current) + len(line) + 1 > max_len and not in_code_block:
                if current:
                    parts.append(current)
                current = line
            else:
                current = (current + '\n' if current else '') + line
        if current:
            parts.append(current)

    for i, part in enumerate(parts):
        try:
            if i == 0 and reply:
                await message.reply(part, parse_mode=ParseMode.HTML)
            else:
                await message.answer(part, parse_mode=ParseMode.HTML)
            if i < len(parts) - 1:
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error sending HTML part {i+1}: {e}")
            try:
                # Fallback: strip all HTML tags
                plain_text = re.sub(r'<[^>]+>', '', part)
                if i == 0 and reply:
                    await message.reply(plain_text, parse_mode=None)
                else:
                    await message.answer(plain_text, parse_mode=None)
            except Exception as ex:
                logger.error(f"Failed to send: {ex}")
            except Exception as ex:
                logger.error(f"Failed to send: {ex}")

class Database:
    def __init__(self, db_name: str = "nova_bot.db"):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute("""CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY, api_key TEXT NOT NULL,
            model TEXT DEFAULT 'gpt-4.1-nano',
            bot_enabled INTEGER DEFAULT 1,
            base_url TEXT DEFAULT 'https://api.proxyapi.ru/openai/v1')""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS admins (
            user_id INTEGER PRIMARY KEY, username TEXT,
            first_name TEXT, last_name TEXT,
            logged_in_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS allowed_groups (
            group_id INTEGER PRIMARY KEY, group_name TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER,
            user_id INTEGER, username TEXT, message TEXT,
            response TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS homework (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS homework_marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            homework_id INTEGER,
            student_id INTEGER,
            completed INTEGER DEFAULT 0,
            marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (homework_id) REFERENCES homework(id),
            FOREIGN KEY (student_id) REFERENCES students(id),
            UNIQUE(homework_id, student_id))""")

        cursor.execute('SELECT COUNT(*) FROM settings')
        if cursor.fetchone()[0] == 0:
            cursor.execute('INSERT INTO settings (id, api_key) VALUES (1, ?)', ('YOUR_API_KEY_HERE',))

        cursor.execute('SELECT COUNT(*) FROM students')
        if cursor.fetchone()[0] == 0:
            initial_students = [
                'Ростислав Ермаков', 'Екатерина Кугутова', 'Владислав Волков',
                'Михаил Морозов', 'Арина Абуева', 'Денис Гармаков',
                'Алиса Малышев', 'Арина Левченко', 'Никита Кузнецов',
                'Алексей Засухин', 'Араик Абрамян'
            ]
            for student in initial_students:
                cursor.execute('INSERT INTO students (name) VALUES (?)', (student,))

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

    def update_base_url(self, url):
        conn = sqlite3.connect(self.db_name)
        conn.execute('UPDATE settings SET base_url=? WHERE id=1', (url,))
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

    def get_students(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT * FROM students ORDER BY name')
        r = c.fetchall()
        conn.close()
        return [{'id': x[0], 'name': x[1], 'added_at': x[2]} for x in r]

    def add_student(self, name):
        conn = sqlite3.connect(self.db_name)
        conn.execute('INSERT INTO students (name) VALUES (?)', (name,))
        conn.commit()
        conn.close()

    def remove_student(self, student_id):
        conn = sqlite3.connect(self.db_name)
        conn.execute('DELETE FROM students WHERE id=?', (student_id,))
        conn.execute('DELETE FROM homework_marks WHERE student_id=?', (student_id,))
        conn.commit()
        conn.close()

    def create_homework(self, title):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('INSERT INTO homework (title) VALUES (?)', (title,))
        hw_id = c.lastrowid
        conn.commit()
        conn.close()
        return hw_id

    def get_homework_list(self, limit=10):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT * FROM homework ORDER BY created_at DESC LIMIT ?', (limit,))
        r = c.fetchall()
        conn.close()
        return [{'id': x[0], 'title': x[1], 'created_at': x[2]} for x in r]

    def get_homework_marks(self, homework_id):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute("""SELECT s.id, s.name, COALESCE(hm.completed, 0)
                     FROM students s
                     LEFT JOIN homework_marks hm ON s.id = hm.student_id AND hm.homework_id = ?
                     ORDER BY s.name""", (homework_id,))
        r = c.fetchall()
        conn.close()
        return [{'student_id': x[0], 'name': x[1], 'completed': bool(x[2])} for x in r]

    def toggle_homework_mark(self, homework_id, student_id):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT completed FROM homework_marks WHERE homework_id=? AND student_id=?', (homework_id, student_id))
        r = c.fetchone()

        if r is None:
            c.execute('INSERT INTO homework_marks (homework_id, student_id, completed) VALUES (?,?,1)', (homework_id, student_id))
        else:
            new_val = 0 if r[0] else 1
            c.execute('UPDATE homework_marks SET completed=?, marked_at=CURRENT_TIMESTAMP WHERE homework_id=? AND student_id=?', 
                     (new_val, homework_id, student_id))

        conn.commit()
        conn.close()

    def delete_homework(self, homework_id):
        conn = sqlite3.connect(self.db_name)
        conn.execute('DELETE FROM homework WHERE id=?', (homework_id,))
        conn.execute('DELETE FROM homework_marks WHERE homework_id=?', (homework_id,))
        conn.commit()
        conn.close()

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
            "content": f"Ты — {BOT_NAME}, дружелюбный AI-ассистент. Отвечай на русском. Используй Markdown: **жирный**, *курсив*, `код`, ```блоки кода``` с языком (```python, ```javascript, ```html и т.д.). Используй заголовки ## и ###. Структурируй ответы."
        }]

        for h in hist:
            messages.append({"role": "user", "content": h['message']})
            messages.append({"role": "assistant", "content": h['response']})

        messages.append({"role": "user", "content": msg})

        try:
            async with aiohttp.ClientSession() as session:
                # Check for GPT-5 or other models requiring Responses API
                if 'gpt-5' in s['model'] or 'nano' in s['model']:
                   # Responses API Implementation
                    data = {
                        "model": s['model'],
                        "input": messages
                    }
                    async with session.post(f"{s['base_url']}/responses",
                                          headers={"Authorization": f"Bearer {s['api_key']}", "Content-Type": "application/json"},
                                          json=data, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        if resp.status == 200:
                            r = await resp.json()
                            # Parse Responses API output
                            output_text = ""
                            for item in r.get('output', []):
                                if item.get('type') == 'message':
                                    for content in item.get('content', []):
                                        if content.get('type') == 'output_text':
                                            output_text += content.get('text', '')
                            return output_text
                        else:
                            error_text = await resp.text()
                            logger.error(f"API error {resp.status}: {error_text}")
                            return f"❌ API ошибка {resp.status} (Responses API): {error_text[:200]}"
                else:
                    # Legacy Chat Completions API
                    data = {"model": s['model'], "messages": messages, "temperature": 0.7}
                    if 'o1' in s['model'] or 'o3' in s['model']:
                         data['max_completion_tokens'] = 2000
                    else:
                         data['max_tokens'] = 2000

                    async with session.post(f"{s['base_url']}/chat/completions",
                                          headers={"Authorization": f"Bearer {s['api_key']}", "Content-Type": "application/json"},
                                          json=data, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status == 200:
                            r = await resp.json()
                            return r['choices'][0]['message']['content']
                        else:
                            error_text = await resp.text()
                            logger.error(f"API error {resp.status}: {error_text}")
                            return f"❌ API ошибка {resp.status}. Проверьте модель."
        except asyncio.TimeoutError:
            return "⏱️ Превышено время ожидания"
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"❌ Ошибка: {str(e)}"

class NovaBot:
    def __init__(self, token):
        self.bot = Bot(token=token, default=DefaultBotProperties())
        self.dp = Dispatcher(storage=MemoryStorage())
        self.db = Database()
        self.ai = NovaAI(self.db)
        self.setup_handlers()

    def setup_handlers(self):
        self.dp.message.register(self.proc_password, ConfigStates.waiting_for_password)
        self.dp.message.register(self.proc_api_key, ConfigStates.waiting_for_api_key)
        self.dp.message.register(self.proc_api_url, ConfigStates.waiting_for_api_url)
        self.dp.message.register(self.proc_model, ConfigStates.waiting_for_model)
        self.dp.message.register(self.proc_broadcast, ConfigStates.waiting_for_broadcast_message)
        self.dp.message.register(self.proc_student_name, ConfigStates.waiting_for_student_name)
        self.dp.message.register(self.proc_homework_title, ConfigStates.waiting_for_homework_title)

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
        self.dp.message.register(self.cmd_broadcast, Command("broadcast"))
        self.dp.message.register(self.cmd_chatid, Command("chatid"))
        self.dp.message.register(self.cmd_journal, Command("journal"))
        self.dp.message.register(self.cmd_help, Command("help"))

        self.dp.my_chat_member.register(self.handle_my_chat_member)

        self.dp.message.register(self.handle_photo, F.photo)
        self.dp.message.register(self.handle_voice, F.voice)
        self.dp.message.register(self.handle_document, F.document)

        self.dp.message.register(self.handle_group_message, F.chat.type.in_({"group", "supergroup"}))
        self.dp.message.register(self.handle_private_message, F.chat.type == "private")

        self.dp.callback_query.register(self.handle_callback)

    async def setup_bot_commands(self):
        commands = [
            BotCommand(command="start", description="🚀 Начать работу с ботом"),
            BotCommand(command="help", description="❓ Помощь и список команд"),
            BotCommand(command="chatid", description="🆔 Узнать ID чата"),
            BotCommand(command="admin", description="⚙️ Админ-панель"),
            BotCommand(command="journal", description="📖 Электронный журнал"),
            BotCommand(command="status", description="📊 Статус бота"),
            BotCommand(command="groups", description="👥 Список групп"),
            BotCommand(command="broadcast", description="📢 Рассылка"),
            BotCommand(command="login", description="🔐 Войти как админ"),
            BotCommand(command="logout", description="👋 Выйти из админки"),
        ]
        await self.bot.set_my_commands(commands)
        logger.info("Bot commands menu set successfully")

    async def handle_my_chat_member(self, update: types.ChatMemberUpdated):
        if update.new_chat_member.status in ["member", "administrator"]:
            if update.chat.type in ["group", "supergroup"]:
                self.db.add_group(update.chat.id, update.chat.title)
                logger.info(f"Bot added to group: {update.chat.title} (ID: {update.chat.id})")
                try:
                    await self.bot.send_message(
                        update.chat.id,
                        f"👋 Привет! Я {BOT_NAME}.\n\n"
                        f"📝 ID этой группы: `{update.chat.id}`\n\n"
                        f"Упомяни меня '{BOT_NAME}' чтобы задать вопрос!",
                        parse_mode=ParseMode.MARKDOWN_V2
                    )
                except:
                    pass

    async def proc_password(self, msg: Message, state: FSMContext):
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

    async def proc_api_key(self, msg: Message, state: FSMContext):
        # Skip if user entered a command instead of value
        if msg.text and msg.text.startswith('/'):
            await state.clear()
            return
        if self.db.is_admin(msg.from_user.id):
            self.db.update_api_key(msg.text.strip())
            await msg.answer("✅ API ключ обновлен")
            await state.clear()
            try:
                await msg.delete()
            except:
                pass

    async def proc_api_url(self, msg: Message, state: FSMContext):
        if msg.text and msg.text.startswith('/'):
            await state.clear()
            return
        if self.db.is_admin(msg.from_user.id):
            self.db.update_base_url(msg.text.strip())
            await msg.answer(f"✅ API URL обновлен: {msg.text.strip()}")
            await state.clear()

    async def proc_model(self, msg: Message, state: FSMContext):
        if msg.text and msg.text.startswith('/'):
            await state.clear()
            return
        if self.db.is_admin(msg.from_user.id):
            self.db.update_model(msg.text.strip())
            await msg.answer(f"✅ Модель изменена на: {msg.text.strip()}\n\n⚠️ Убедитесь что эта модель поддерживается вашим API!")
            await state.clear()

    async def proc_broadcast(self, msg: Message, state: FSMContext):
        if msg.text and msg.text.startswith('/'):
            await state.clear()
            return
        if self.db.is_admin(msg.from_user.id):
            await self.broadcast_message(msg.text, msg)
            await state.clear()

    async def proc_student_name(self, msg: Message, state: FSMContext):
        if msg.text and msg.text.startswith('/'):
            await state.clear()
            return
        if self.db.is_admin(msg.from_user.id):
            self.db.add_student(msg.text.strip())
            await msg.answer(f"✅ Студент '{msg.text.strip()}' добавлен!")
            await state.clear()

    async def proc_homework_title(self, msg: Message, state: FSMContext):
        if msg.text and msg.text.startswith('/'):
            await state.clear()
            return
        if self.db.is_admin(msg.from_user.id):
            hw_id = self.db.create_homework(msg.text.strip())
            await msg.answer(f"✅ ДЗ создано: {msg.text.strip()}")
            await self.show_homework_marks(msg, hw_id)
            await state.clear()

    async def cmd_start(self, msg):
        admin_txt = "\n\n🔐 Вы админ. /admin" if self.db.is_admin(msg.from_user.id) else "\n\n🔐 Доступ: /login"
        text = (f"👋 Привет! Я {BOT_NAME} — AI-ассистент.\n\n"
                f"📝 В группе: упомяни '{BOT_NAME}'\n"
                f"💬 Личка: пиши напрямую\n"
                f"🎤 Голосовые сообщения\n"
                f"📄 Файлы (.txt, .py и т.д.)\n"
                f"🖼️ Фотографии{admin_txt}\n\n"
                f"Используй /help для списка команд")
        await send_markdown_message(msg, text)

    async def cmd_help(self, msg):
        is_admin = self.db.is_admin(msg.from_user.id)

        text = f"📚 **Команды {BOT_NAME}**\n\n"
        text += "**Основные:**\n"
        text += "/start - Начать работу\n"
        text += "/help - Эта справка\n"
        text += "/chatid - Узнать ID чата\n\n"

        if is_admin:
            text += "**Администрирование:**\n"
            text += "/admin - Админ-панель\n"
            text += "/journal - Электронный журнал\n"
            text += "/status - Статус бота\n"
            text += "/groups - Список групп\n"
            text += "/admins - Список админов\n"
            text += "/broadcast - Рассылка\n"
            text += "/addgroup - Добавить группу\n"
            text += "/removegroup - Удалить группу\n"
            text += "/setapi - Установить API ключ\n"
            text += "/setmodel - Сменить модель\n"
            text += "/toggle - Вкл/Выкл бота\n"
            text += "/logout - Выйти из админки\n\n"
        else:
            text += "**Вход:**\n"
            text += "/login - Войти как админ\n\n"

        text += "**Возможности:**\n"
        text += "🤖 Общение с AI (GPT)\n"
        text += "🖼️ Анализ изображений\n"
        text += "🎤 Распознавание речи\n"
        text += "📁 Анализ файлов\n"

        if is_admin:
            text += "📖 Электронный журнал\n"
            text += "📊 Статистика и отчеты"

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

    async def cmd_chatid(self, msg):
        chat_type = "👤 Личный чат" if msg.chat.type == "private" else f"👥 Группа: {msg.chat.title}"
        is_allowed = "✅ Разрешен" if self.db.is_group_allowed(msg.chat.id) else "❌ Не разрешен"
        if msg.chat.type in ["group", "supergroup"]:
            text = f"📊 **Информация о чате:**\n\n{chat_type}\nID: `{msg.chat.id}`\nСтатус: {is_allowed}\n\nДля добавления: /addgroup"
        else:
            text = f"📊 **Информация о чате:**\n\n{chat_type}\nВаш ID: `{msg.from_user.id}`"
        await send_markdown_message(msg, text)

    async def cmd_admin(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            await msg.answer("⛔ Только админ")
            return
        s = self.db.get_settings()
        status = "🟢 Вкл" if s['bot_enabled'] else "🔴 Выкл"
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Вкл/Выкл", callback_data="toggle_bot")],
            [InlineKeyboardButton(text="🔑 API Ключ", callback_data="change_api"),
             InlineKeyboardButton(text="🌐 API URL", callback_data="change_url")],
            [InlineKeyboardButton(text="🤖 Модель", callback_data="change_model")],
            [InlineKeyboardButton(text="📊 Статус", callback_data="show_status")],
            [InlineKeyboardButton(text="📢 Рассылка", callback_data="broadcast")],
            [InlineKeyboardButton(text="📖 Журнал", callback_data="journal_menu")],
            [InlineKeyboardButton(text="👥 Группы", callback_data="manage_groups"),
             InlineKeyboardButton(text="🔐 Админы", callback_data="show_admins")]
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
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="❌ Отмена", callback_data="show_admin")]
            ])
            await msg.answer("🔑 Отправьте API ключ:", reply_markup=kb)
            await state.set_state(ConfigStates.waiting_for_api_key)

    async def cmd_setmodel(self, msg, state):
        if not self.db.is_admin(msg.from_user.id):
            return
        args = msg.text.split(maxsplit=1)
        if len(args) > 1:
            self.db.update_model(args[1].strip())
            await msg.answer(f"✅ Модель: {args[1].strip()}")
        else:
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="❌ Отмена", callback_data="show_admin")]
            ])
            await msg.answer("🤖 Отправьте модель (например: gpt-4o-mini):", reply_markup=kb)
            await state.set_state(ConfigStates.waiting_for_model)

    async def cmd_toggle(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        new = self.db.toggle_bot()
        await msg.answer(f"Бот {'🟢 Вкл' if new else '🔴 Выкл'}")

    async def cmd_status(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        await self.show_status_view(msg)

    async def show_status_view(self, msg):
        s = self.db.get_settings()
        g = self.db.get_allowed_groups()
        a = self.db.get_admins()
        students = self.db.get_students()
        homeworks = self.db.get_homework_list(limit=100)

        text = f"📊 **Статус {BOT_NAME}**\n\n"
        text += f"🤖 Состояние: {'🟢 Включен' if s['bot_enabled'] else '🔴 Выключен'}\n"
        text += f"🧠 Модель: {s['model']}\n"
        text += f"🌐 API URL: {s['base_url']}\n"
        text += f"🔑 API ключ: {'✅ Установлен' if s['api_key'] != 'YOUR_API_KEY_HERE' else '❌ Не установлен'}\n\n"
        text += f"📊 **Статистика:**\n"
        text += f"👥 Групп: {len(g)}\n"
        text += f"🔐 Админов: {len(a)}\n"
        text += f"👨‍🎓 Студентов: {len(students)}\n"
        text += f"📚 Домашних заданий: {len(homeworks)}\n"

        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="← Назад", callback_data="show_admin")]
        ])
        await send_markdown_message(msg, text)
        await msg.answer("Меню:", reply_markup=kb)

    async def cmd_addgroup(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        if msg.chat.type in ["group", "supergroup"]:
            self.db.add_group(msg.chat.id, msg.chat.title)
            await msg.answer(f"✅ Группа '{msg.chat.title}' добавлена\nID: `{msg.chat.id}`", parse_mode=ParseMode.MARKDOWN_V2)
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
                await msg.answer("📝 Используйте в группе или: /addgroup -100...\n💡 Совет: /chatid для ID", parse_mode=ParseMode.MARKDOWN_V2)

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
        await self.show_groups_menu(msg)

    async def show_groups_menu(self, msg):
        g = self.db.get_allowed_groups()
        if not g:
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="← Назад", callback_data="show_admin")]
            ])
            await msg.answer("📝 Нет групп", reply_markup=kb)
            return
        buttons = []
        for x in g:
            name = x['group_name'] or 'Без названия'
            buttons.append([InlineKeyboardButton(text=f"🗑️ {name}", callback_data=f"del_group_{x['group_id']}")])
        buttons.append([InlineKeyboardButton(text="← Назад", callback_data="show_admin")])
        kb = InlineKeyboardMarkup(inline_keyboard=buttons)
        await msg.answer("👥 **Группы** (нажмите для удаления):", reply_markup=kb, parse_mode=ParseMode.MARKDOWN)

    async def cmd_admins(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        await self.show_admins_view(msg)

    async def show_admins_view(self, msg):
        a = self.db.get_admins()
        if not a:
            await msg.answer("📝 Нет админов")
            return
        text = "**Администраторы:**\n\n"
        for x in a:
            name = x['first_name'] or 'Без имени'
            uname = f"@{x['username']}" if x['username'] else 'без username'
            text += f"• {name} ({uname})\n  ID: `{x['user_id']}`\n"
        
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="← Назад", callback_data="show_admin")]
        ])
        await send_markdown_message(msg, text)
        await msg.answer("Меню:", reply_markup=kb)

    async def cmd_broadcast(self, msg, state):
        if not self.db.is_admin(msg.from_user.id):
            await msg.answer("⛔ Только админ")
            return
        args = msg.text.split(maxsplit=1)
        if len(args) > 1:
            await self.broadcast_message(args[1], msg)
        else:
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="❌ Отмена", callback_data="show_admin")]
            ])
            await msg.answer("📢 Отправьте сообщение для рассылки во все группы:", reply_markup=kb)
            await state.set_state(ConfigStates.waiting_for_broadcast_message)

    async def broadcast_message(self, text, source_msg):
        groups = self.db.get_allowed_groups()
        if not groups:
            await source_msg.answer("❌ Нет доступных групп для рассылки")
            return
        success = 0
        failed = 0
        failed_groups = []
        await source_msg.answer(f"📤 Начинаю рассылку в {len(groups)} групп(ы)...")
        for group in groups:
            try:
                await self.bot.send_message(group['group_id'], text, parse_mode=None)
                success += 1
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to send to {group['group_id']}: {e}")
                failed += 1
                failed_groups.append(f"{group['group_name'] or 'Без названия'} (ID: {group['group_id']})")
        result_text = f"✅ Рассылка завершена!\n\n📊 Успешно: {success}\n❌ Ошибок: {failed}"
        if failed_groups:
            result_text += f"\n\n⚠️ Не удалось отправить в:\n" + "\n".join(f"• {g}" for g in failed_groups[:5])
            if len(failed_groups) > 5:
                result_text += f"\n...и ещё {len(failed_groups)-5}"
        await source_msg.answer(result_text)

    async def cmd_journal(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            await msg.answer("⛔ Только админ")
            return
        await self.show_journal_menu(msg)

    async def show_journal_menu(self, msg):
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="➕ Создать ДЗ", callback_data="journal_create_hw")],
            [InlineKeyboardButton(text="📝 Список ДЗ", callback_data="journal_hw_list")],
            [InlineKeyboardButton(text="👥 Студенты", callback_data="journal_students")],
            [InlineKeyboardButton(text="← Назад", callback_data="show_admin")]
        ])
        await msg.answer("📖 *Электронный журнал*", reply_markup=kb, parse_mode=ParseMode.MARKDOWN)

    async def show_homework_list(self, msg):
        homeworks = self.db.get_homework_list(limit=10)
        if not homeworks:
            await msg.answer("📝 Нет домашних заданий")
            return
        buttons = []
        for hw in homeworks:
            date = hw['created_at'][:10] if hw['created_at'] else ''
            buttons.append([InlineKeyboardButton(text=f"📚 {hw['title']} ({date})", callback_data=f"journal_hw_{hw['id']}")])
        buttons.append([InlineKeyboardButton(text="← Назад", callback_data="journal_menu")])
        kb = InlineKeyboardMarkup(inline_keyboard=buttons)
        await msg.answer("📝 *Домашние задания:*", reply_markup=kb, parse_mode=ParseMode.MARKDOWN)

    async def show_homework_marks(self, msg, homework_id: int):
        hw_list = self.db.get_homework_list(limit=100)
        hw = next((h for h in hw_list if h['id'] == homework_id), None)
        if not hw:
            await msg.answer("❌ ДЗ не найдено")
            return
        marks = self.db.get_homework_marks(homework_id)
        buttons = []
        for mark in marks:
            emoji = "✅" if mark['completed'] else "❌"
            buttons.append([InlineKeyboardButton(text=f"{emoji} {mark['name']}", callback_data=f"journal_mark_{homework_id}_{mark['student_id']}")])
        buttons.append([InlineKeyboardButton(text="📢 Отправить отчет", callback_data=f"journal_send_{homework_id}")])
        buttons.append([InlineKeyboardButton(text="🗑️ Удалить ДЗ", callback_data=f"journal_delete_{homework_id}")])
        buttons.append([InlineKeyboardButton(text="← Назад", callback_data="journal_hw_list")])
        kb = InlineKeyboardMarkup(inline_keyboard=buttons)
        completed = sum(1 for m in marks if m['completed'])
        total = len(marks)
        percentage = (completed / total * 100) if total > 0 else 0
        text = f"📚 *{hw['title']}*\n\n📊 Статистика: {completed}/{total} ({percentage:.0f}%)\n\nНажми на имя студента чтобы изменить статус"
        await msg.answer(text, reply_markup=kb, parse_mode=ParseMode.MARKDOWN)

    async def send_homework_report(self, homework_id: int, source_msg):
        hw_list = self.db.get_homework_list(limit=100)
        hw = next((h for h in hw_list if h['id'] == homework_id), None)
        if not hw:
            await source_msg.answer("❌ ДЗ не найдено")
            return
        marks = self.db.get_homework_marks(homework_id)
        completed = [m for m in marks if m['completed']]
        not_completed = [m for m in marks if not m['completed']]
        date = datetime.now().strftime("%d.%m.%Y %H:%M")
        report = f"📚 *Отчет по домашнему заданию*\n\n*Тема:* {hw['title']}\n*Дата:* {date}\n\n"
        if completed:
            report += f"✅ *Сделали ({len(completed)}):*\n"
            for m in completed:
                report += f"• {m['name']}\n"
            report += "\n"
        if not_completed:
            report += f"❌ *Не сделали ({len(not_completed)}):*\n"
            for m in not_completed:
                report += f"• {m['name']}\n"
            report += "\n"
        total = len(marks)
        percentage = (len(completed) / total * 100) if total > 0 else 0
        report += f"📈 *Статистика:* {percentage:.0f}% ({len(completed)} из {total})"
        groups = self.db.get_allowed_groups()
        success = 0
        failed = 0
        for group in groups:
            try:
                await self.bot.send_message(group['group_id'], report, parse_mode=ParseMode.MARKDOWN)
                success += 1
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to send report to {group['group_id']}: {e}")
                failed += 1
        await source_msg.answer(f"✅ Отчет отправлен!\n\n📊 Успешно: {success}\n❌ Ошибок: {failed}")

    async def show_students_menu(self, msg):
        students = self.db.get_students()
        text = "👥 **Студенты** ({} чел.)\n\n".format(len(students))
        for i, s in enumerate(students, 1):
            text += f"{i}. {s['name']}\n"
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="➕ Добавить студента", callback_data="journal_add_student")],
            [InlineKeyboardButton(text="❌ Удалить студента", callback_data="journal_remove_student")],
            [InlineKeyboardButton(text="← Назад", callback_data="journal_menu")]
        ])
        await msg.answer(text, reply_markup=kb, parse_mode=ParseMode.MARKDOWN)

    async def show_students_for_removal(self, msg):
        students = self.db.get_students()
        if not students:
            await msg.answer("📝 Нет студентов")
            return
        buttons = []
        for student in students:
            buttons.append([InlineKeyboardButton(text=f"🗑️ {student['name']}", callback_data=f"journal_confirm_del_{student['id']}")])
        buttons.append([InlineKeyboardButton(text="← Назад", callback_data="journal_students")])
        kb = InlineKeyboardMarkup(inline_keyboard=buttons)
        await msg.answer("👥 **Выберите студента для удаления:**", reply_markup=kb, parse_mode=ParseMode.MARKDOWN)

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
                self.db.save_chat_history(msg.chat.id, msg.from_user.id, msg.from_user.username or msg.from_user.first_name, f"[Фото]: {msg.caption or 'нет'}", analysis)
                await send_markdown_message(msg, f"🖼️ **Анализ изображения:**\n\n{analysis}", reply=True)
            else:
                await msg.reply("❌ Ошибка анализа")
        except Exception as e:
            logger.error(f"Photo error: {e}")
            await msg.reply(f"❌ Error: {e}")

    async def cmd_status(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        await self.show_status_view(msg)

    async def show_status_view(self, msg):
        s = self.db.get_settings()
        g = self.db.get_allowed_groups()
        a = self.db.get_admins()
        students = self.db.get_students()
        homeworks = self.db.get_homework_list(limit=100)

        text = f"📊 **Статус {BOT_NAME}**\n\n"
        text += f"🤖 Состояние: {'🟢 Включен' if s['bot_enabled'] else '🔴 Выключен'}\n"
        text += f"🧠 Модель: {s['model']}\n"
        text += f"🌐 API URL: {s['base_url']}\n"
        text += f"🔑 API ключ: {'✅ Установлен' if s['api_key'] != 'YOUR_API_KEY_HERE' else '❌ Не установлен'}\n\n"
        text += f"📊 **Статистика:**\n"
        text += f"👥 Групп: {len(g)}\n"
        text += f"🔐 Админов: {len(a)}\n"
        text += f"👨‍🎓 Студентов: {len(students)}\n"
        text += f"📚 Домашних заданий: {len(homeworks)}\n"

        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="← Назад", callback_data="show_admin")]
        ])
        await send_markdown_message(msg, text)
        await msg.answer("Меню:", reply_markup=kb)

    async def cmd_addgroup(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        if msg.chat.type in ["group", "supergroup"]:
            self.db.add_group(msg.chat.id, msg.chat.title)
            await msg.answer(f"✅ Группа '{msg.chat.title}' добавлена\nID: `{msg.chat.id}`", parse_mode=ParseMode.HTML)
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
                await msg.answer("📝 Используйте в группе или: /addgroup -100...\n💡 Совет: /chatid для ID", parse_mode=ParseMode.HTML)

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
        await self.show_groups_menu(msg)

    async def show_groups_menu(self, msg):
        g = self.db.get_allowed_groups()
        if not g:
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="← Назад", callback_data="show_admin")]
            ])
            await msg.answer("📝 Нет групп", reply_markup=kb)
            return
        buttons = []
        for x in g:
            name = x['group_name'] or 'Без названия'
            buttons.append([InlineKeyboardButton(text=f"🗑️ {name}", callback_data=f"del_group_{x['group_id']}")])
        buttons.append([InlineKeyboardButton(text="← Назад", callback_data="show_admin")])
        kb = InlineKeyboardMarkup(inline_keyboard=buttons)
        await msg.answer("👥 **Группы** (нажмите для удаления):", reply_markup=kb, parse_mode=ParseMode.HTML)

    async def cmd_admins(self, msg):
        if not self.db.is_admin(msg.from_user.id):
            return
        await self.show_admins_view(msg)

    async def show_admins_view(self, msg):
        a = self.db.get_admins()
        if not a:
            await msg.answer("📝 Нет админов")
            return
        text = "<b>Администраторы:</b>\n\n"
        for x in a:
            name = x['first_name'] or 'Без имени'
            uname = f"@{x['username']}" if x['username'] else 'без username'
            text += f"• {name} ({uname})\n  ID: <code>{x['user_id']}</code>\n"
        
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="← Назад", callback_data="show_admin")]
        ])
        await msg.answer(text, reply_markup=kb, parse_mode=ParseMode.HTML)

    async def cmd_broadcast(self, msg, state):
        if not self.db.is_admin(msg.from_user.id):
            await msg.answer("⛔ Только админ")
            return
        args = msg.text.split(maxsplit=1)
        if len(args) > 1:
            await self.broadcast_message(args[1], msg)
        else:
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="❌ Отмена", callback_data="show_admin")]
            ])
            await msg.answer("📢 Отправьте текст сообщения для рассылки:", reply_markup=kb)
            await state.set_state(ConfigStates.waiting_for_broadcast_message)

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
                self.db.save_chat_history(msg.chat.id, msg.from_user.id, msg.from_user.username or msg.from_user.first_name, f"[Голос]: {trans}", resp)
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
            exts = ['.txt', '.md', '.py', '.js', '.html', '.css', '.java', '.cpp', '.cs', '.sql', '.php', '.go', '.rs', '.csv']
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
            self.db.save_chat_history(msg.chat.id, msg.from_user.id, msg.from_user.username or msg.from_user.first_name, f"[Файл]: {doc.file_name}", resp)
            await send_markdown_message(msg, f"📁 **Анализ файла:** {doc.file_name}\n\n{resp}", reply=True)
        except Exception as e:
            logger.error(f"Doc error: {e}")
            await msg.reply(f"❌ Error: {e}")

    async def handle_group_message(self, msg):
        s = self.db.get_settings()
        if not s['bot_enabled'] or not self.db.is_group_allowed(msg.chat.id):
            return
        
        # Check if this is a reply to bot's message
        is_reply_to_bot = False
        if msg.reply_to_message and msg.reply_to_message.from_user:
            is_reply_to_bot = msg.reply_to_message.from_user.id == self.bot.id
        
        # Check if bot is mentioned by name
        is_mentioned = msg.text and BOT_NAME.lower() in msg.text.lower()
        
        # Only respond if mentioned or replied to
        if not is_mentioned and not is_reply_to_bot:
            return
        
        if not msg.text:
            return
            
        umsg = msg.text
        # Remove bot name from message if mentioned
        if is_mentioned:
            for v in [BOT_NAME, BOT_NAME.lower(), BOT_NAME.upper()]:
                umsg = umsg.replace(v, "").strip()
        
        if not umsg:
            umsg = "Привет!"
        await msg.chat.do("typing")
        resp = await self.ai.generate_response(umsg, msg.chat.id)
        self.db.save_chat_history(msg.chat.id, msg.from_user.id, msg.from_user.username or msg.from_user.first_name, umsg, resp)
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
        self.db.save_chat_history(msg.chat.id, msg.from_user.id, msg.from_user.username or msg.from_user.first_name, msg.text, resp)
        await send_markdown_message(msg, resp)

    async def handle_callback(self, cb, state):
        if not self.db.is_admin(cb.from_user.id):
            await cb.answer("⛔ Только админ", show_alert=True)
            return

        if cb.data == "cancel_action":
            await state.clear()
            await cb.message.answer("❌ Действие отменено")
            await cb.answer()
            return

        if cb.data == "journal_menu":
            await self.show_journal_menu(cb.message)
            await cb.answer()
        elif cb.data == "journal_create_hw":
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="❌ Отмена", callback_data="journal_menu")]])
            await cb.message.answer("📝 Введите название домашнего задания:", reply_markup=kb)
            await state.set_state(ConfigStates.waiting_for_homework_title)
            await cb.answer()
        elif cb.data == "journal_hw_list":
            await self.show_homework_list(cb.message)
            await cb.answer()
        elif cb.data.startswith("journal_hw_"):
            hw_id = int(cb.data.split("_")[2])
            await self.show_homework_marks(cb.message, hw_id)
            await cb.answer()
        elif cb.data.startswith("journal_mark_"):
            parts = cb.data.split("_")
            hw_id = int(parts[2])
            student_id = int(parts[3])
            self.db.toggle_homework_mark(hw_id, student_id)
            await self.show_homework_marks(cb.message, hw_id)
            await cb.answer("✅ Отметка изменена")
        elif cb.data.startswith("journal_send_"):
            hw_id = int(cb.data.split("_")[2])
            await self.send_homework_report(hw_id, cb.message)
            await cb.answer()
        elif cb.data.startswith("journal_delete_"):
            hw_id = int(cb.data.split("_")[2])
            self.db.delete_homework(hw_id)
            await cb.message.answer("🗑️ ДЗ удалено")
            await self.show_homework_list(cb.message)
            await cb.answer()
        elif cb.data == "journal_students":
            await self.show_students_menu(cb.message)
            await cb.answer()
        elif cb.data == "journal_add_student":
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="❌ Отмена", callback_data="journal_students")]])
            await cb.message.answer("👤 Введите ФИО студента:", reply_markup=kb)
            await state.set_state(ConfigStates.waiting_for_student_name)
            await cb.answer()
        elif cb.data == "journal_remove_student":
            await self.show_students_for_removal(cb.message)
            await cb.answer()
        elif cb.data.startswith("journal_confirm_del_"):
            student_id = int(cb.data.split("_")[3])
            self.db.remove_student(student_id)
            await cb.message.answer("✅ Студент удален")
            await self.show_students_for_removal(cb.message)
            await cb.answer()
        elif cb.data == "show_admin":
            await self.cmd_admin(cb.message)
            await cb.answer()
        elif cb.data == "toggle_bot":
            new = self.db.toggle_bot()
            await cb.answer(f"Бот {'🟢 Вкл' if new else '🔴 Выкл'}")
            await self.cmd_admin(cb.message)
        elif cb.data == "change_api":
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="❌ Отмена", callback_data="show_admin")]])
            await cb.message.answer("🔑 Отправьте API ключ:", reply_markup=kb)
            await state.set_state(ConfigStates.waiting_for_api_key)
            await cb.answer()
        elif cb.data == "change_url":
            s = self.db.get_settings()
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="❌ Отмена", callback_data="show_admin")]])
            await cb.message.answer(f"🌐 Текущий URL: {s['base_url']}\n\nОтправьте новый API URL:", reply_markup=kb)
            await state.set_state(ConfigStates.waiting_for_api_url)
            await cb.answer()
        elif cb.data == "change_model":
            s = self.db.get_settings()
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="❌ Отмена", callback_data="show_admin")]])
            await cb.message.answer(f"🤖 Текущая модель: {s['model']}\n\nОтправьте название модели:", reply_markup=kb)
            await state.set_state(ConfigStates.waiting_for_model)
            await cb.answer()
        elif cb.data == "broadcast":
            kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="❌ Отмена", callback_data="show_admin")]])
            await cb.message.answer("📢 Отправьте текст сообщения для рассылки:", reply_markup=kb)
            await state.set_state(ConfigStates.waiting_for_broadcast_message)
            await cb.answer()
        elif cb.data == "show_status":
            try:
                logger.info("Executing show_status_view")
                await self.show_status_view(cb.message)
            except Exception as e:
                logger.error(f"Failed to show status: {e}", exc_info=True)
                await cb.answer(f"Ошибка: {e}", show_alert=True)
                return
            await cb.answer()
        elif cb.data == "manage_groups":
            await self.show_groups_menu(cb.message)
            await cb.answer()
        elif cb.data.startswith("del_group_"):
            group_id = int(cb.data.split("_")[2])
            self.db.remove_group(group_id)
            await cb.answer("✅ Группа удалена")
            await self.show_groups_menu(cb.message)
        elif cb.data == "show_admins":
            try:
                logger.info("Executing show_admins_view")
                await self.show_admins_view(cb.message)
            except Exception as e:
                logger.error(f"Failed to show admins: {e}", exc_info=True)
                await cb.answer(f"Ошибка: {e}", show_alert=True)
                return
            await cb.answer()

    async def start(self):
        await self.setup_bot_commands()
        logger.info(f"Starting {BOT_NAME}...")
        await self.dp.start_polling(self.bot)

async def main():
    TOKEN = os.getenv("BOT_TOKEN")
    if not TOKEN:
        logger.error("BOT_TOKEN environment variable is not set!")
        logger.error("Please create a .env file with BOT_TOKEN=your_token")
        return
    bot = NovaBot(TOKEN)
    await bot.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped")
