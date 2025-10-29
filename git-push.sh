#!/bin/bash

echo "🚀 Подготовка проекта для Git..."
echo ""

read -p "Введите URL вашего Git репозитория (например, https://github.com/username/repo.git): " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ URL репозитория не указан!"
    exit 1
fi

echo ""
echo "📦 Инициализация Git репозитория..."
git init

echo ""
echo "➕ Добавление файлов..."
git add .

echo ""
echo "💾 Создание первого коммита..."
git commit -m "Initial commit: Ball physics simulation project

Проект моделирования движения шара:
- Веб-интерфейс (Streamlit)
- Консольная версия
- 9 юнит-тестов
- Полная физическая документация

Реализованные сценарии:
- Скатывание по наклонной
- Проскальзывание
- Качение по горизонтали
- Столкновения со стенами
- Несколько шаров"

echo ""
echo "🔗 Подключение к удаленному репозиторию..."
git remote add origin "$REPO_URL"

echo ""
echo "⬆️ Отправка в репозиторий..."
git branch -M main
git push -u origin main

echo ""
echo "✅ Готово! Проект загружен в репозиторий:"
echo "   $REPO_URL"
echo ""
echo "📋 Теперь вы можете:"
echo "   - Открыть репозиторий в браузере"
echo "   - Добавить описание проекта"
echo "   - Настроить GitHub Pages (если нужно)"

