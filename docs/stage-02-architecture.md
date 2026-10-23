# Отчёт по этапу 2: проектирование архитектуры и каркас приложения

## 1. Цель этапа

Создать запускаемый скелет проекта с конфигурацией, базовыми модулями, FastAPI-приложением и логированием.

## 2. Выполненные задачи

- [x] Создана модульная структура `backend`.
- [x] Добавлены модули `backend/evaluation/` и `backend/data/api_llm.py`.
- [x] Реализован `backend/core/config.py` на базе `pydantic-settings`.
- [x] Добавлены настройки БД, LLM, Qdrant, Redis, JWT и внешнего LLM API.
- [x] Созданы `backend/core/constants.py` и `backend/core/exceptions.py`.
- [x] Создано FastAPI-приложение в `backend/api/main.py`.
- [x] Подключён CORS.
- [x] Добавлен endpoint `GET /healthz`.
- [x] Создан `run.py` для запуска через Uvicorn.
- [x] Настроено логирование в консоль и файл `logs/app.log`.
- [x] Добавлено вращение файла логов при достижении 5 МБ.

## 3. Структура основных модулей

```text
backend/
├── agents/
├── api/
│   ├── main.py
│   └── routes/
├── core/
│   ├── config.py
│   ├── constants.py
│   └── exceptions.py
├── data/
│   └── api_llm.py
├── db/
├── evaluation/
├── hitl/
├── retrieval/
├── utils/
│   └── logging.py
└── worker/
```

## 4. Проверка запуска

Локальный запуск:

```powershell
poetry run python run.py
```

Проверка endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

Ожидаемый результат:

```json
{
  "status": "ok",
  "service": "KPFU Web Assistant",
  "version": "0.1.0"
}
```

## 5. Результат этапа

Приложение имеет запускаемый FastAPI-каркас, конфигурация читается из переменных окружения, а события записываются одновременно в консоль и файл.

Следующий этап — подключение базы данных и создание моделей хранения документов, вопросов, ответов и логов.
