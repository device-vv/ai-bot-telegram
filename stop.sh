#!/bin/bash

# Файл, где хранится PID
PID_FILE="bot.pid"

echo "Останавливаем бота..."

# Проверяем, существует ли файл с PID
if [ -f "$PID_FILE" ]; then
    # Считываем PID из файла
    PID=$(cat $PID_FILE)

    # Убиваем процесс по его PID
    kill $PID

    # Удаляем файл с PID
    rm $PID_FILE

    echo "Бот с PID $PID остановлен."
else
    echo "Файл $PID_FILE не найден. Возможно, бот не запущен?"
fi
