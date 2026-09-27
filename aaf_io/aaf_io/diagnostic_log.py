"""Explicit categories for repetitive diagnostics on the string callback API."""
from __future__ import annotations

from threading import RLock
from typing import Callable


class DiagnosticMessage(str):
    """A string-compatible per-item warning; ordinary status lines stay unbounded."""

    category: str

    def __new__(cls, message: str, *, category: str):
        value = super().__new__(cls, message)
        value.category = category
        return value


class BoundedDiagnosticLog:
    """Own per-category budgets for one synchronous operation, including its workers.

    Only explicitly categorized per-item diagnostics are limited. Call finish()
    after producers stop, including on failure/cancellation. Status and final
    errors remain ordinary strings and are always forwarded.
    """

    def __init__(self, emit: Callable[[str], None], *, limit: int = 10):
        if limit < 0:
            raise ValueError('Diagnostic limit must be nonnegative')
        self._emit = emit
        self._limit = limit
        self._counts: dict[str, int] = {}
        self._lock = RLock()
        self._finished = False

    def __call__(self, message: str) -> None:
        with self._lock:
            if self._finished:
                raise RuntimeError('Diagnostic log already finished')
            if not isinstance(message, DiagnosticMessage):
                self._emit(message)
                return
            category = message.category
            count = self._counts.get(category, 0) + 1
            self._counts[category] = count
            if count <= self._limit:
                self._emit(str(message))
            elif count == self._limit + 1:
                self._emit(
                    f'Лог [{category}]: показано {self._limit} сообщений; '
                    'дальнейшие сообщения этой категории скрываются.'
                )

    def finish(self) -> None:
        """Emit accurate suppression totals once; never reset a live budget."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            for category, count in self._counts.items():
                if count > self._limit:
                    self._emit(
                        f'Лог [{category}]: скрыто {count - self._limit} '
                        f'сообщений (всего {count}, показано {self._limit}).'
                    )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()
        return False
