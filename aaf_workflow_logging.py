from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol


class StageProgress(Protocol):
    def set_stage(self, value: str) -> None: ...

    def set_indeterminate(self, value: bool) -> None: ...


@dataclass(frozen=True)
class StageLogger:
    emit: Optional[Callable[[str], None]] = None
    progress: Optional[StageProgress] = None

    def start(
        self,
        message: str,
        *,
        stage: Optional[str] = None,
        indeterminate: Optional[bool] = None,
    ) -> None:
        if self.progress is not None:
            if indeterminate is not None:
                self.progress.set_indeterminate(bool(indeterminate))
            if stage is not None:
                self.progress.set_stage(str(stage))
        if self.emit is not None:
            self.emit(str(message))

    def detail(self, message: str) -> None:
        if self.emit is not None:
            self.emit(str(message))

    def done(self, message: str) -> None:
        if self.emit is not None:
            self.emit(str(message))
