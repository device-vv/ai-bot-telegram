# 🤖 Nova Bot

**[English](#english) | [Русский](#русский)**

---

## English

AI-powered Telegram assistant with an electronic gradebook feature.

### ✨ Features

- 💬 **AI Chat** — GPT-based conversations in private messages and groups
- 🖼️ **Image Analysis** — Image descriptions via GPT-4 Vision
- 🎤 **Speech Recognition** — Voice message transcription (Whisper)
- 📄 **File Analysis** — Supports `.txt`, `.md`, `.csv` and code (`.py`, `.js`, `.java`, `.cpp`, `.sql`, `.html`, etc.)
- 📖 **Electronic Gradebook** — Homework and student tracking
- 📢 **Broadcasting** — Send messages to all groups

### 🚀 Installation

#### 1. Clone the repository

```bash
git clone https://github.com/your-username/nova-bot.git
cd nova-bot
```

#### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit the `.env` file:

```env
BOT_TOKEN=your_telegram_bot_token      # Get from @BotFather
ADMIN_PASSWORD=your_secure_password    # Admin panel password
OPENAI_API_KEY=your_api_key            # OpenAI API key
API_BASE_URL=https://api.openai.com/v1 # API URL (optional)
AI_MODEL=gpt-5-nano                  # AI model (optional)
```

#### 5. Run the bot

**Manual run:**
```bash
python3 nova_bot.py
```

**Using scripts (Background mode):**
```bash
# Start the bot in background
chmod +x start.sh stop.sh
./start.sh

# Stop the bot
./stop.sh
```

### 📋 Commands

#### Basic Commands
| Command | Description |
|---------|-------------|
| `/start` | Start the bot |
| `/help` | List of commands |
| `/chatid` | Get chat ID |

#### Admin Commands (requires authentication)
| Command | Description |
|---------|-------------|
| `/login` | Login as administrator |
| `/logout` | Logout from admin panel |
| `/admin` | Open admin panel |
| `/journal` | Electronic gradebook |
| `/status` | Bot status |
| `/broadcast` | Broadcast to all groups |
| `/groups` | List of allowed groups |
| `/admins` | List of administrators |
| `/toggle` | Enable/disable bot |
| `/setapi` | Set API key |
| `/setmodel` | Change AI model |

### 🔐 Security

- Bot token and passwords are stored in `.env` file
- `.env` is added to `.gitignore` and **never committed**
- Database (`nova_bot.db`) is also excluded from the repository

### 🛠️ Technologies

- [aiogram 3](https://docs.aiogram.dev/) — Telegram Bot Framework
- [aiohttp](https://docs.aiohttp.org/) — HTTP client
- [python-dotenv](https://github.com/theskumar/python-dotenv) — Environment variable loading
- SQLite — Database

---

## Русский

AI-ассистент для Telegram с функциями электронного журнала.

### ✨ Возможности

- 💬 **AI-чат** — общение с GPT в личных сообщениях и группах
- 🖼️ **Анализ изображений** — описание картинок через GPT-4 Vision
- 🎤 **Распознавание речи** — транскрипция голосовых сообщений (Whisper)
- 📄 **Анализ файлов** — поддержка `.txt`, `.md`, `.csv` и кода (`.py`, `.js`, `.java`, `.cpp`, `.sql`, `.html` и др.)
- 📖 **Электронный журнал** — учёт домашних заданий и студентов
- 📢 **Рассылка** — отправка сообщений во все группы

### 🚀 Установка

#### 1. Клонируйте репозиторий

```bash
git clone https://github.com/your-username/nova-bot.git
cd nova-bot
```

#### 2. Создайте виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate     # Windows
```

#### 3. Установите зависимости

```bash
pip install -r requirements.txt
```

#### 4. Настройте переменные окружения

```bash
cp .env.example .env
```

Отредактируйте `.env` файл:

```env
BOT_TOKEN=your_telegram_bot_token      # Получить у @BotFather
ADMIN_PASSWORD=your_secure_password    # Пароль для доступа к админке
OPENAI_API_KEY=your_api_key            # API ключ OpenAI
API_BASE_URL=https://api.openai.com/v1 # URL API (опционально)
AI_MODEL=gpt-5-nano                  # Модель AI (опционально)
```

#### 5. Запустите бота

**Вручную:**
```bash
python3 nova_bot.py
```

**С помощью скриптов (Фоновый режим):**
```bash
# Запустить бота в фоне
chmod +x start.sh stop.sh
./start.sh

# Остановить бота
./stop.sh
```

### 📋 Команды

#### Основные
| Команда | Описание |
|---------|----------|
| `/start` | Начать работу с ботом |
| `/help` | Список команд |
| `/chatid` | Узнать ID чата |

#### Администрирование (требуется авторизация)
| Команда | Описание |
|---------|----------|
| `/login` | Войти как администратор |
| `/logout` | Выйти из админки |
| `/admin` | Открыть админ-панель |
| `/journal` | Электронный журнал |
| `/status` | Статус бота |
| `/broadcast` | Рассылка во все группы |
| `/groups` | Список разрешённых групп |
| `/admins` | Список администраторов |
| `/toggle` | Включить/выключить бота |
| `/setapi` | Установить API ключ |
| `/setmodel` | Сменить модель AI |

### 🔐 Безопасность

- Токен бота и пароли хранятся в `.env` файле
- Файл `.env` добавлен в `.gitignore` и **не коммитится**
- База данных (`nova_bot.db`) также исключена из репозитория

### 📁 Структура проекта

```
nova-bot/
├── nova_bot.py       # Основной код бота
├── requirements.txt  # Зависимости Python
├── start.sh          # Скрипт запуска
├── stop.sh           # Скрипт остановки
├── .env.example      # Шаблон переменных окружения
├── .gitignore        # Исключения для Git
├── LICENSE           # Лицензия MIT
└── README.md         # Документация
```

### 🛠️ Технологии

- [aiogram 3](https://docs.aiogram.dev/) — Telegram Bot Framework
- [aiohttp](https://docs.aiohttp.org/) — HTTP клиент
- [python-dotenv](https://github.com/theskumar/python-dotenv) — Загрузка переменных окружения
- SQLite — База данных

---

## 📝 License / Лицензия

MIT License
