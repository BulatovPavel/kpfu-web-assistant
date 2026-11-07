"""
Unit tests for backend.db.models.

Uses SQLite in-memory so no Postgres is required. Each test creates
a fresh engine and drops it afterwards, so tests are fully isolated.
"""

import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.db.models import GlossaryDB, TranslationCache, UncertainExample, User, Base
import pytest_asyncio

@pytest.fixture
def engine():
    """One fresh in-memory engine per test."""
    e = create_async_engine("sqlite+aiosqlite:///:memory:")
    yield e
    asyncio.run(e.dispose())


@pytest.fixture
async def session(engine):
    """Create all tables and yield a session."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as s:
        yield s


class TestTablesCreated:
    def test_all_five_tables(self):
        expected = {
            "users",
            "knowledge_base",
            "glossaries",
            "uncertain_examples",
            "cache",
            "evaluation_results",
            "model_comparisons",
            "reasoning_logs",
            "llm_calls",
        }
        assert set(Base.metadata.tables.keys()) == expected

    def test_glossary_has_langs_index(self):
        idx_names = {i.name for i in GlossaryDB.__table__.indexes}
        assert "ix_glossaries_langs" in idx_names

    # def test_tm_has_langs_index(self):
    #     idx_names = {i.name for i in TMDB.__table__.indexes}
    #     assert "ix_tm_langs" in idx_names

    def test_cache_has_eval_mode_index(self):
        idx_names = {i.name for i in TranslationCache.__table__.indexes}
        assert "ix_cache_eval_mode" in idx_names


class TestUserDefaults:
    async def test_user_insert(self, session):
        u = User(
            email="a@b.com",
            username="alice",
            hashed_password="hash",
        )
        session.add(u)
        await session.commit()

        found = (await session.execute(
            select(User).where(User.email == "a@b.com")
        )).scalar_one()
        assert found.username == "alice"
        assert found.role == "user"
        assert found.is_active is True


# class TestTMDBRoundTrip:
#     async def test_insert_and_select(self, session):
#         entry = TMDB(
#             id="en|ru|abcdef",
#             source_lang="en",
#             target_lang="ru",
#             source_text="hello",
#             target_text="привет",
#         )
#         session.add(entry)
#         await session.commit()

#         found = (await session.execute(
#             select(TMDB).where(TMDB.id == "en|ru|abcdef")
#         )).scalar_one()
#         assert found.source_text == "hello"
#         assert found.target_text == "привет"

#     async def test_deterministic_id_is_primary_key(self, session):
#         """Two inserts with the same id — the second fails on PK conflict."""
#         first = TMDB(
#             id="en|ru|same",
#             source_lang="en", target_lang="ru",
#             source_text="hello", target_text="привет",
#         )
#         session.add(first)
#         await session.commit()

#         second = TMDB(
#             id="en|ru|same",
#             source_lang="en", target_lang="ru",
#             source_text="hi", target_text="здравствуй",
#         )
#         session.add(second)
#         with pytest.raises(Exception):  # IntegrityError
#             await session.commit()
#         await session.rollback()


class TestCacheSentinel:
    async def test_cache_defaults_to_unknown(self, session):
        c = TranslationCache(
            source_text="hi",
            source_lang="en", target_lang="ru",
            translation="привет",
        )
        session.add(c)
        await session.commit()
        await session.refresh(c)

        assert c.engine_used == "unknown"
        assert c.eval_mode == "unknown"
        assert c.model_used == "unknown"
        assert c.hits == 1


class TestUncertainExample:
    async def test_pending_by_default(self, session):
        ex = UncertainExample(
            source_text="hi", source_lang="en", target_lang="ru",
            machine_translation="привет",
            critic_score=5.0,
            validation_passed=False,
        )
        session.add(ex)
        await session.commit()
        await session.refresh(ex)

        assert ex.status == "pending"
        assert ex.corrected_translation is None
        assert ex.confirmed_at is None