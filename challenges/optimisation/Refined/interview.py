"""Asking the user the things the lattice cannot answer, reproducibly.

`describe.py` is a conversation: it shows what it found and asks what it means. That
is the right shape — the judgements are genuinely human — but an interactive script
is also one that cannot be tested, cannot be re-run in CI, and leaves no record of
why a machine was described the way it was.

So every question has an identifier, and every answer is recorded. The same run can
be replayed from a file, scripted with `--defaults`, or audited afterwards. The
interview is the interface; the answer file is the artifact.

    interview = Interview(answers=replayed, defaults_only=False)
    kind = interview.choose("vbc.role", "Is this string a knob or geometry?",
                            ["knob", "fixed"], default="fixed",
                            help="A compressor is a knob; a dogleg is geometry.")
    interview.save("answers.json")
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

RULE = "─" * 72


@dataclass
class Interview:
    """A recorded question-and-answer session with a human, or a replay of one."""

    answers: dict[str, Any] = field(default_factory=dict)
    # Take the default for anything not already answered, and ask nothing. For
    # scripts, tests, and a first look at an unfamiliar lattice.
    defaults_only: bool = False
    # Everything asked, in order, with what was given and where it came from.
    transcript: list[dict[str, Any]] = field(default_factory=list)

    # ---------- the primitives ----------

    def _resolve(self, key: str, default, render: Callable[[Any], str] | None = None):
        """An answer from the replay file, or None if it has to be asked for."""
        if key in self.answers:
            value = self.answers[key]
            self._record(key, value, "replayed")
            return value
        if self.defaults_only:
            self._record(key, default, "default")
            return default
        return None

    def _record(self, key: str, value, source: str) -> None:
        self.answers[key] = value
        self.transcript.append({"key": key, "value": value, "source": source})

    def _prompt(self, text: str, help: str | None, suffix: str) -> str:
        if help:
            for line in help.strip().splitlines():
                print(f"    {line.strip()}")
        try:
            return input(f"  {text} {suffix} ").strip()
        except EOFError:
            print("  (no input available — taking the default)")
            return ""

    # ---------- the question types ----------

    def choose(self, key: str, text: str, options: Sequence[str],
               default: str, help: str | None = None) -> str:
        replay = self._resolve(key, default)
        if replay is not None:
            if replay not in options:
                raise ValueError(f"{key}: {replay!r} is not one of {list(options)}")
            return replay
        shown = "/".join(f"[{o}]" if o == default else o for o in options)
        while True:
            given = self._prompt(text, help, f"({shown}):") or default
            if given in options:
                self._record(key, given, "asked")
                return given
            print(f"      not one of {list(options)}")

    def confirm(self, key: str, text: str, default: bool = True,
                help: str | None = None) -> bool:
        replay = self._resolve(key, default)
        if replay is not None:
            return bool(replay)
        given = self._prompt(text, help, "[Y/n]:" if default else "[y/N]:")
        value = default if not given else given.lower().startswith("y")
        self._record(key, value, "asked")
        return value

    def text(self, key: str, prompt: str, default: str = "",
             help: str | None = None, allow_empty: bool = True) -> str:
        replay = self._resolve(key, default)
        if replay is not None:
            return str(replay)
        while True:
            given = self._prompt(prompt, help, f"[{default}]:" if default else ":")
            value = given or default
            if value or allow_empty:
                self._record(key, value, "asked")
                return value
            print("      a value is needed here")

    def number(self, key: str, prompt: str, default: float,
               help: str | None = None, cast=float,
               bounds: tuple[float, float] | None = None):
        replay = self._resolve(key, default)
        if replay is not None:
            return cast(replay)
        while True:
            given = self._prompt(prompt, help, f"[{default:g}]:")
            if not given:
                self._record(key, cast(default), "asked")
                return cast(default)
            try:
                value = cast(given)
            except ValueError:
                print("      not a number")
                continue
            if bounds and not (bounds[0] <= value <= bounds[1]):
                print(f"      outside {bounds}")
                continue
            self._record(key, value, "asked")
            return value

    # ---------- presentation ----------

    @staticmethod
    def section(title: str) -> None:
        print(f"\n{RULE}\n  {title}\n{RULE}")

    @staticmethod
    def say(*lines: str) -> None:
        for line in lines:
            print(f"  {line}")

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"answers": self.answers, "transcript": self.transcript}, indent=1)
        )

    @classmethod
    def replay(cls, path: str | Path | None, defaults_only: bool = False) -> "Interview":
        if path is None:
            return cls(defaults_only=defaults_only)
        blob = json.loads(Path(path).read_text())
        return cls(answers=dict(blob.get("answers", blob)), defaults_only=defaults_only)

    @property
    def asked(self) -> int:
        return sum(1 for t in self.transcript if t["source"] == "asked")
