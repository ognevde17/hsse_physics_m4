
echo "Запуск проекта моделирования движения шара..."
echo ""

if ! command -v python3 &> /dev/null; then
    echo "Python 3 не найден! Установите Python 3.8 или новее."
    exit 1
fi

if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Установка зависимостей..."
    pip3 install -r requirements.txt
    echo ""
fi

echo "Запуск веб-интерфейса..."
echo "Откроется браузер: http://localhost:8501"
echo ""
echo "Для остановки нажмите Ctrl+C"
echo ""

python3 -m streamlit run app.py

