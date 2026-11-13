import pytest

from backend.agents.supervisor import Supervisor


class FakeValidator:
    def __init__(self, validation_results):
        self.validation_results = iter(validation_results)
        self.last_validation_result = validation_results[-1]
        self.calls = 0
        self._last_result = None

    async def process(self, state):
        self.calls += 1
        try:
            passed = next(self.validation_results)
            self.last_validation_result = passed
        except StopIteration:
            passed = self.last_validation_result

        state["validation_passed"] = passed
        state["validator_errors"] = [] if passed else ["needs correction"]
        state["validator_warnings"] = []
        self._last_result = {"passed": passed}
        return state

    def get_last_result(self):
        return self._last_result


class FakeCritic:
    def __init__(self, scores, failed_results=None, suggestions=None):
        self.scores = iter(scores)
        self.last_score = scores[-1]
        self.failed_results = iter(failed_results or [False])
        self.last_failed = False
        self.suggestions = suggestions or []
        self.calls = 0
        self._last_result = None

    async def process(self, state):
        self.calls += 1
        try:
            score = next(self.scores)
            self.last_score = score
        except StopIteration:
            score = self.last_score
        try:
            failed = next(self.failed_results)
            self.last_failed = failed
        except StopIteration:
            failed = self.last_failed

        state["critic_score"] = score
        state["critic_failed"] = failed
        state["critic_suggestions"] = list(self.suggestions)
        self._last_result = {"overall_score": score, "failed": failed}
        return state

    def get_last_result(self):
        return self._last_result


class FakeEditor:
    def __init__(self, *, changed=True, failed=False, corrected_answer="Updated answer"):
        self.changed = changed
        self.failed = failed
        self.corrected_answer = corrected_answer
        self.calls = 0
        self._last_result = None

    async def process(self, state):
        self.calls += 1
        state["editor_changed"] = self.changed
        state["editor_failed"] = self.failed
        if self.changed:
            state["answer"] = self.corrected_answer
        self._last_result = {"changed": self.changed, "failed": self.failed}
        return state

    def get_last_result(self):
        return self._last_result


def install_agents(monkeypatch, supervisor, validator, critic=None, editor=None):
    monkeypatch.setattr(supervisor, "_get_validator", lambda: validator)
    if critic is not None:
        monkeypatch.setattr(supervisor, "_get_critic", lambda: critic)
    if editor is not None:
        monkeypatch.setattr(supervisor, "_get_editor", lambda: editor)


@pytest.mark.asyncio
async def test_agents_disabled_runs_only_validator(monkeypatch):
    supervisor = Supervisor()
    validator = FakeValidator([True])
    critic = FakeCritic([8.0])
    editor = FakeEditor()
    install_agents(monkeypatch, supervisor, validator, critic, editor)

    result = await supervisor.process({
        "answer": "Answer",
        "use_critic": False,
        "use_editor": False,
    })

    assert validator.calls == 1
    assert critic.calls == 0
    assert editor.calls == 0
    assert result["supervisor_stopped_reason"] == "agents_disabled"


@pytest.mark.asyncio
async def test_critic_failed_stops_before_editor(monkeypatch):
    supervisor = Supervisor()
    validator = FakeValidator([True])
    critic = FakeCritic([None], failed_results=[True])
    editor = FakeEditor()
    install_agents(monkeypatch, supervisor, validator, critic, editor)

    result = await supervisor.process({"answer": "Answer"})

    assert validator.calls == 1
    assert critic.calls == 1
    assert editor.calls == 0
    assert result["supervisor_stopped_reason"] == "critic_failed"


@pytest.mark.asyncio
async def test_editor_disabled_stops_after_critic(monkeypatch):
    supervisor = Supervisor()
    validator = FakeValidator([True])
    critic = FakeCritic([8.0])
    editor = FakeEditor()
    install_agents(monkeypatch, supervisor, validator, critic, editor)

    result = await supervisor.process({
        "answer": "Answer",
        "use_critic": True,
        "use_editor": False,
    })

    assert validator.calls == 1
    assert critic.calls == 1
    assert editor.calls == 0
    assert result["supervisor_stopped_reason"] == "editor_disabled"


@pytest.mark.asyncio
async def test_converged_with_high_score_skips_editor(monkeypatch):
    supervisor = Supervisor()
    validator = FakeValidator([True])
    critic = FakeCritic([8.0])
    editor = FakeEditor()
    install_agents(monkeypatch, supervisor, validator, critic, editor)

    result = await supervisor.process({"answer": "Answer"})

    assert result["supervisor_iterations"] == 0
    assert result["supervisor_stopped_reason"] == "converged"
    assert editor.calls == 0


@pytest.mark.asyncio
async def test_validation_failure_limits_editor_to_one_iteration(monkeypatch):
    supervisor = Supervisor(max_iterations=3)
    validator = FakeValidator([False, True])
    critic = FakeCritic([8.0, 8.0], suggestions=["clarify one detail"])
    editor = FakeEditor()
    install_agents(monkeypatch, supervisor, validator, critic, editor)

    result = await supervisor.process({"answer": "Answer"})

    assert result["supervisor_budget"] == 1
    assert result["supervisor_iterations"] == 1
    assert validator.calls == 2
    assert critic.calls == 2
    assert editor.calls == 1
    assert result["answer"] == "Updated answer"


@pytest.mark.asyncio
async def test_rollback_regression_restores_previous_state(monkeypatch):
    supervisor = Supervisor()
    validator = FakeValidator([True, True])
    critic = FakeCritic([8.0, 5.0], suggestions=["address a concern"])
    editor = FakeEditor(corrected_answer="Worse answer")
    install_agents(monkeypatch, supervisor, validator, critic, editor)

    result = await supervisor.process({"answer": "Original answer"})

    assert result["answer"] == "Original answer"
    assert result["critic_score"] == 8.0
    assert result["editor_rolled_back"] is True
    assert result["supervisor_stopped_reason"] == "rollback_regression"


@pytest.mark.asyncio
async def test_editor_failure_rolls_back(monkeypatch):
    supervisor = Supervisor()
    validator = FakeValidator([False, False])
    critic = FakeCritic([5.0, 5.0])
    editor = FakeEditor(failed=True)
    install_agents(monkeypatch, supervisor, validator, critic, editor)

    result = await supervisor.process({"answer": "Original answer"})

    assert result["answer"] == "Original answer"
    assert result["editor_rolled_back"] is True
    assert result["supervisor_stopped_reason"] == "rollback_error"


@pytest.mark.asyncio
async def test_editor_no_change_stops_iteration(monkeypatch):
    supervisor = Supervisor()
    validator = FakeValidator([False, False])
    critic = FakeCritic([5.0, 5.0])
    editor = FakeEditor(changed=False)
    install_agents(monkeypatch, supervisor, validator, critic, editor)

    result = await supervisor.process({"answer": "Original answer"})

    assert editor.calls == 1
    assert result["supervisor_iterations"] == 1
    assert result["editor_rolled_back"] is False
    assert result["supervisor_stopped_reason"] == "no_change"


def test_snapshot_restore_recovers_list_after_mutation():
    state = {"items": ["before"], "answer": "Original"}
    snapshot = Supervisor._snapshot(state)
    state["items"].append("after")
    state["answer"] = "Changed"

    Supervisor._restore(state, snapshot)

    assert state == {"items": ["before"], "answer": "Original"}
