"""Schema contract tests for dispatch_agent and dispatch_chat."""

from hermes_herald import tools

_SECRET_WARNING = (
    "Avoid placing secrets or credentials in the task text — "
    "the dispatch ledger stores the full message as supplied."
)


def test_dispatch_agent_schema_warns_against_secrets_in_task_text():
    properties = tools.DISPATCH_AGENT_SCHEMA["parameters"]["properties"]
    assert _SECRET_WARNING in properties["message"]["description"]
    assert _SECRET_WARNING in properties["instructions"]["description"]


def test_dispatch_chat_schema_warns_against_secrets_in_task_text():
    properties = tools.DISPATCH_CHAT_SCHEMA["parameters"]["properties"]
    assert _SECRET_WARNING in properties["message"]["description"]
    assert _SECRET_WARNING in properties["instructions"]["description"]
