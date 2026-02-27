"""Project-scoped file operation layer with safety boundaries.

This module provides read, write, edit, and soft-delete operations constrained
to project and allowlisted roots. It also enforces sensitive path denylist
rules and returns operation records suitable for state/delta integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable
import shutil


DEFAULT_SENSITIVE_PATTERNS: tuple[str, ...] = (
    ".env",
    "*.pem",
    "*.key",
    "secrets.*",
)


class FileOperationError(Exception):
    """Base exception for file operation failures."""


class BoundaryViolationError(FileOperationError):
    """Raised when a path is outside project or allowlisted boundaries."""


class SensitivePathError(FileOperationError):
    """Raised when a path matches a sensitive denylist pattern."""


@dataclass(frozen=True)
class FileOperationRecord:
    """Record describing a filesystem operation.

    Attributes:
        op_type: Operation type for state/delta usage.
        target_path: Absolute path of operation target.
        payload: Operation payload details.
        inverse_payload: Reverse operation details when reversible.
        reversible: Whether operation can be reversed.
        created_at: UTC timestamp of operation record creation.
    """

    op_type: str
    target_path: str
    payload: dict[str, Any]
    inverse_payload: dict[str, Any] | None
    reversible: bool
    created_at: str


class FileOperationLayer:
    """Safe file operation layer constrained to project boundaries."""

    def __init__(
        self,
        project_root: Path,
        allowlist_roots: Iterable[Path] | None = None,
        sensitive_patterns: tuple[str, ...] = DEFAULT_SENSITIVE_PATTERNS,
        trash_dir_name: str = ".luna9_trash",
    ) -> None:
        """Initialize project-scoped file operations.

        Args:
            project_root: Root path for allowed project operations.
            allowlist_roots: Additional absolute roots allowed for operations.
            sensitive_patterns: Glob patterns disallowed for access.
            trash_dir_name: Soft-delete directory name under project root.
        """
        self.project_root = project_root.resolve()
        self.allowlist_roots = tuple(root.resolve() for root in (allowlist_roots or []))
        self.sensitive_patterns = sensitive_patterns
        self.trash_root = self.project_root / trash_dir_name

    def read_text(self, path: str | Path) -> tuple[str, FileOperationRecord]:
        """Read text file content within allowed boundaries.

        Args:
            path: Relative or absolute file path.

        Returns:
            Tuple containing file text and an operation record.

        Raises:
            FileOperationError: If file cannot be read safely.
        """
        resolved = self._validate_path(path)
        if not resolved.exists() or not resolved.is_file():
            raise FileOperationError(f"Path is not a file: {resolved}")

        content = resolved.read_text(encoding="utf-8")
        record = FileOperationRecord(
            op_type="file_read",
            target_path=str(resolved),
            payload={"bytes": len(content.encode("utf-8"))},
            inverse_payload=None,
            reversible=False,
            created_at=self._utc_now_iso(),
        )
        return content, record

    def write_text(
        self,
        path: str | Path,
        content: str,
        create_parents: bool = True,
    ) -> FileOperationRecord:
        """Write text content to a file within boundaries.

        Args:
            path: Relative or absolute file path.
            content: Text content to write.
            create_parents: Whether to create missing parent directories.

        Returns:
            Operation record describing file creation or modification.
        """
        resolved = self._validate_path(path)
        if create_parents:
            resolved.parent.mkdir(parents=True, exist_ok=True)

        existed = resolved.exists()
        previous = ""
        if existed:
            previous = resolved.read_text(encoding="utf-8")

        resolved.write_text(content, encoding="utf-8")
        op_type = "file_modified" if existed else "file_created"
        inverse_payload: dict[str, Any] | None
        if existed:
            inverse_payload = {"content": previous}
        else:
            inverse_payload = {"delete_path": str(resolved)}

        return FileOperationRecord(
            op_type=op_type,
            target_path=str(resolved),
            payload={"content": content},
            inverse_payload=inverse_payload,
            reversible=True,
            created_at=self._utc_now_iso(),
        )

    def edit_text(self, path: str | Path, new_content: str) -> FileOperationRecord:
        """Edit an existing file by replacing its full text content.

        Args:
            path: Relative or absolute file path.
            new_content: New full content for the target file.

        Returns:
            Operation record describing the modification.
        """
        resolved = self._validate_path(path)
        if not resolved.exists() or not resolved.is_file():
            raise FileOperationError(f"Path is not a file: {resolved}")

        previous = resolved.read_text(encoding="utf-8")
        resolved.write_text(new_content, encoding="utf-8")
        return FileOperationRecord(
            op_type="file_modified",
            target_path=str(resolved),
            payload={"content": new_content},
            inverse_payload={"content": previous},
            reversible=True,
            created_at=self._utc_now_iso(),
        )

    def soft_delete(self, path: str | Path) -> FileOperationRecord:
        """Soft-delete a file by moving it to project trash storage.

        Args:
            path: Relative or absolute file path.

        Returns:
            Operation record containing original and trash paths.
        """
        resolved = self._validate_path(path)
        if not resolved.exists():
            raise FileOperationError(f"Path does not exist: {resolved}")

        relative = resolved.relative_to(self.project_root)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        trash_target = self.trash_root / timestamp / relative
        trash_target.parent.mkdir(parents=True, exist_ok=True)
        self.trash_root.mkdir(parents=True, exist_ok=True)

        shutil.move(str(resolved), str(trash_target))
        return FileOperationRecord(
            op_type="file_soft_deleted",
            target_path=str(resolved),
            payload={"trash_path": str(trash_target)},
            inverse_payload={
                "restore_from": str(trash_target),
                "restore_to": str(resolved),
            },
            reversible=True,
            created_at=self._utc_now_iso(),
        )

    def _validate_path(self, path: str | Path) -> Path:
        """Resolve and validate a path against boundary and sensitivity rules."""
        candidate = Path(path)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.project_root / candidate).resolve()
        )

        if not self._is_allowed(resolved):
            raise BoundaryViolationError(f"Path is outside allowed roots: {resolved}")

        if self._is_sensitive(resolved):
            raise SensitivePathError(f"Path matches sensitive pattern: {resolved}")

        return resolved

    def _is_allowed(self, resolved: Path) -> bool:
        """Return True when path is under project root or allowlisted roots."""
        all_roots = (self.project_root, *self.allowlist_roots)
        for root in all_roots:
            if self._is_relative_to(resolved, root):
                return True
        return False

    def _is_sensitive(self, resolved: Path) -> bool:
        """Return True when path matches denylist patterns."""
        basename = resolved.name
        as_posix = resolved.as_posix()
        for pattern in self.sensitive_patterns:
            if fnmatch(basename, pattern) or fnmatch(as_posix, pattern):
                return True
        return False

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        """Return True when path is located under root."""
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    @staticmethod
    def _utc_now_iso() -> str:
        """Return current UTC timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat()
