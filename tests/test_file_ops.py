"""Tests for project-scoped file operation boundaries."""

from pathlib import Path

import pytest

from luna9.storage.file_ops import (
    BoundaryViolationError,
    FileOperationLayer,
    FileOperationRecord,
    SensitivePathError,
)


def test_write_read_and_edit_within_project_root(tmp_path: Path) -> None:
    """Verify write, read, and edit are allowed under project root."""
    layer = FileOperationLayer(project_root=tmp_path)

    write_record = layer.write_text("notes/todo.txt", "alpha")
    assert isinstance(write_record, FileOperationRecord)
    assert write_record.op_type == "file_created"

    content, read_record = layer.read_text("notes/todo.txt")
    assert content == "alpha"
    assert read_record.op_type == "file_read"

    edit_record = layer.edit_text("notes/todo.txt", "beta")
    assert edit_record.op_type == "file_modified"

    updated, _ = layer.read_text("notes/todo.txt")
    assert updated == "beta"


def test_boundary_violation_for_parent_escape(tmp_path: Path) -> None:
    """Verify paths outside project root are blocked."""
    layer = FileOperationLayer(project_root=tmp_path)

    with pytest.raises(BoundaryViolationError):
        layer.write_text("../outside.txt", "nope")


def test_sensitive_paths_are_blocked(tmp_path: Path) -> None:
    """Verify sensitive denylist patterns prevent access."""
    layer = FileOperationLayer(project_root=tmp_path)

    with pytest.raises(SensitivePathError):
        layer.write_text(".env", "SECRET=1")


def test_allowlisted_root_is_permitted(tmp_path: Path) -> None:
    """Verify allowlisted roots permit controlled external access."""
    allow_root = tmp_path / "allow"
    allow_root.mkdir(parents=True, exist_ok=True)

    layer = FileOperationLayer(
        project_root=tmp_path / "project",
        allowlist_roots=(allow_root,),
    )

    record = layer.write_text(allow_root / "shared.txt", "ok")
    assert record.op_type == "file_created"


def test_soft_delete_moves_to_project_trash(tmp_path: Path) -> None:
    """Verify soft delete moves files under .luna9_trash."""
    layer = FileOperationLayer(project_root=tmp_path)
    target = tmp_path / "work" / "artifact.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("content", encoding="utf-8")

    record = layer.soft_delete(target)

    assert record.op_type == "file_soft_deleted"
    assert target.exists() is False

    trash_path = Path(str(record.payload["trash_path"]))
    assert trash_path.exists() is True
    assert ".luna9_trash" in trash_path.as_posix()
