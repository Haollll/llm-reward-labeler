from pathlib import Path

TASK_DIR = Path(__file__).parent / "tasks"


def load_task(name: str) -> str:
    """Load task description from TASK_DIR."""
    return (TASK_DIR / f"{name}.txt").read_text().strip()


def section(title: str, width: int = 60) -> str:
    """For pretty printing."""
    pad = max(width - len(title) - 4, 4)
    return f"\n\033[1;36m── {title} {'─' * pad}\033[0m"
