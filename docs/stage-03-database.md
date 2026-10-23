# Отчёт по этапу 3: база данных

## 1. Цель этапа

Настроить асинхронное подключение к базе данных, описать ORM-модели и обеспечить автоматическое создание таблиц при запуске приложения.

## 2. Выполненные задачи

- [x] Добавлены SQLAlchemy, `aiosqlite` и `asyncpg`.
- [x] Реализовано асинхронное подключение в `backend/db/session.py`.
- [x] Создана декларативная база `Base`.
- [x] Созданы модели `users`, `knowledge_base`, `glossary`, `uncertain_examples`, `cache`.
- [x] Созданы модели `evaluation_results` и `model_comparisons`.
- [x] Созданы модели `reasoning_logs` и `llm_calls` для трассировки агентов и вызовов LLM.
- [x] Добавлено автоматическое создание таблиц в lifecycle FastAPI-приложения.
- [x] Добавлен PostgreSQL в `infrastructure/docker-compose.yml`.
- [x] Добавлен тест создания всех обязательных таблиц в SQLite.

## 3. Локальная база данных

Основной локальный вариант — PostgreSQL в Docker Compose:

```text
postgresql+asyncpg://kpfu:kpfu_local_password@localhost:5432/kpfu_assistant
```

Запуск PostgreSQL:

```powershell
docker compose -f infrastructure/docker-compose.yml up -d postgres
```

После запуска в `.env` используется:

```text
DATABASE_URL=postgresql+asyncpg://kpfu:kpfu_local_password@localhost:5432/kpfu_assistant
```

SQLite оставлен только для изолированных тестов и не используется обычным локальным запуском приложения.

## 4. Проверки

```powershell
poetry run pytest tests/test_database.py
```

Тест проверяет наличие всех обязательных таблиц, включая `reasoning_logs` и `llm_calls`.

## 5. Результат этапа

Приложение получает асинхронную фабрику сессий, автоматически создаёт схему базы данных при старте и поддерживает локальную SQLite-базу и PostgreSQL через Docker.

Следующий этап — регистрация пользователей, JWT-аутентификация и роли доступа.
