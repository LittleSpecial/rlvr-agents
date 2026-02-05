from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal


EnvType = Literal["code", "sql"]
Variant = Literal["paper_a", "paper_b"]


@dataclass(frozen=True)
class ToyTask:
    task_id: str
    env_type: EnvType
    task: Dict[str, Any]  # passed to env.reset()
    candidates: List[str]  # candidate code / SQL strings (length K)


def get_toy_tasks(env_type: EnvType, *, variant: Variant) -> List[ToyTask]:
    if env_type == "code":
        return _toy_code_tasks(variant=variant)
    if env_type == "sql":
        return _toy_sql_tasks(variant=variant)
    raise ValueError(f"Unknown env_type: {env_type}")


def _toy_code_tasks(*, variant: Variant) -> List[ToyTask]:
    # K=3 candidates for every task; Paper B uses two task types that disagree on which index is correct.
    tasks: List[ToyTask] = []

    def _mk_task(
        task_id: str,
        prompt: str,
        initial_code: str,
        test_code: str,
        candidates: List[str],
        task_type: str,
    ) -> ToyTask:
        return ToyTask(
            task_id=task_id,
            env_type="code",
            task={
                "task_id": task_id,
                "prompt": prompt,
                "initial_code": initial_code,
                "test_code": test_code,
                "timeout": 5.0,
                "metadata": {"task_type": task_type},
            },
            candidates=candidates,
        )

    # Task 1: add
    prompt = "Implement add(a, b) that returns a + b."
    initial = "def add(a, b):\n    return 0\n"
    tests = "assert add(1,2)==3\nassert add(-1,1)==0\n"
    correct = "def add(a, b):\n    return a + b\n"
    wrong0 = "def add(a, b):\n    return 0\n"
    wrong1 = "def add(a, b):\n    return a - b\n"

    if variant == "paper_a":
        tasks.append(_mk_task("code_add", prompt, initial, tests, [correct, wrong0, wrong1], "type0"))
    else:
        # Paper B: mix task types with different correct index to create conflict
        tasks.append(_mk_task("code_add_t0", prompt, initial, tests, [correct, wrong0, wrong1], "type0"))
        tasks.append(_mk_task("code_add_t1", prompt, initial, tests, [wrong0, correct, wrong1], "type1"))

    # Task 2: factorial
    prompt = "Implement fact(n) that returns n! for n>=0."
    initial = "def fact(n):\n    return n\n"
    tests = "assert fact(0)==1\nassert fact(5)==120\n"
    correct = "def fact(n):\n    out = 1\n    for i in range(2, n+1):\n        out *= i\n    return out\n"
    wrong0 = "def fact(n):\n    return n\n"
    wrong1 = "def fact(n):\n    out = 1\n    for i in range(1, n):\n        out *= i\n    return out\n"

    if variant == "paper_a":
        tasks.append(_mk_task("code_fact", prompt, initial, tests, [correct, wrong0, wrong1], "type0"))
    else:
        tasks.append(_mk_task("code_fact_t0", prompt, initial, tests, [correct, wrong0, wrong1], "type0"))
        tasks.append(_mk_task("code_fact_t1", prompt, initial, tests, [wrong0, correct, wrong1], "type1"))

    # Add imbalance for Paper B to encourage collapse: more type0 tasks than type1 tasks.
    if variant == "paper_b":
        tasks.append(_mk_task("code_add_t0b", prompt, initial, tests, [correct, wrong0, wrong1], "type0"))
        tasks.append(_mk_task("code_fact_t0b", prompt, initial, tests, [correct, wrong0, wrong1], "type0"))

    return tasks


def _toy_sql_tasks(*, variant: Variant) -> List[ToyTask]:
    # Minimal SQL tasks; mainly for smoke tests of env/verifier.
    # K=3 candidates for every task; Paper B variant keeps the same type split idea.
    schema = (
        "CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INTEGER);\n"
        "INSERT INTO users(id, name, age) VALUES (1, 'alice', 30);\n"
        "INSERT INTO users(id, name, age) VALUES (2, 'bob', 20);\n"
        "INSERT INTO users(id, name, age) VALUES (3, 'cathy', 40);\n"
    )
    prompt = "Select names of users with age >= 30."
    expected = [("alice",), ("cathy",)]

    correct = "SELECT name FROM users WHERE age >= 30;"
    wrong0 = "SELECT name FROM users WHERE age > 30;"
    wrong1 = "SELECT id FROM users WHERE age >= 30;"

    def _mk(task_id: str, candidates: List[str], task_type: str) -> ToyTask:
        return ToyTask(
            task_id=task_id,
            env_type="sql",
            task={
                "task_id": task_id,
                "prompt": prompt,
                "database_schema": schema,
                "expected_result": expected,
                "metadata": {"task_type": task_type},
            },
            candidates=candidates,
        )

    if variant == "paper_a":
        return [_mk("sql_users", [correct, wrong0, wrong1], "type0")]

    return [
        _mk("sql_users_t0", [correct, wrong0, wrong1], "type0"),
        _mk("sql_users_t1", [wrong0, correct, wrong1], "type1"),
        _mk("sql_users_t0b", [correct, wrong0, wrong1], "type0"),
    ]

