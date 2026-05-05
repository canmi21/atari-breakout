"""Append-only CSV logger for training metrics.

Schema is fixed on the first call to `log()` -- subsequent rows must use
the same keys. Flushes every row so a kill -9 mid-training still leaves
a readable file for the matplotlib post-mortem.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
	def __init__(self, path: Path):
		self.path = path
		self._fields: list[str] | None = None
		self._fh = None
		self._writer: csv.DictWriter | None = None

	def log(self, **row: Any) -> None:
		if self._writer is None:
			self.path.parent.mkdir(parents=True, exist_ok=True)
			self._fh = self.path.open("w", newline="", encoding="utf-8")
			self._fields = list(row.keys())
			self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
			self._writer.writeheader()
		self._writer.writerow(row)
		assert self._fh is not None
		self._fh.flush()

	def close(self) -> None:
		if self._fh is not None:
			self._fh.close()
			self._fh = None
			self._writer = None

	def __enter__(self) -> CSVLogger:
		return self

	def __exit__(self, *_exc: object) -> None:
		self.close()
