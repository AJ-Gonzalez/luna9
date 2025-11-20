"""
Pytest configuration and fixtures for Luna Nine tests.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import pytest

# Add luna9-python to path
sys.path.insert(0, str(Path(__file__).parent.parent / "luna9-python"))

from luna9.domain_manager import DomainManager


@pytest.fixture
def temp_storage():
    """Provide temporary storage directory for tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix="luna9_test_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def dm(temp_storage):
    """Provide DomainManager instance with temporary storage."""
    return DomainManager(storage_dir=temp_storage)


@pytest.fixture
def dm_with_hierarchy(dm):
    """Provide DomainManager with pre-created hierarchy."""
    dm.create_domain("personal", "PERSONAL")
    dm.create_domain("foundation", "FOUNDATION")
    dm.create_domain("foundation/books", "FOUNDATION")
    dm.create_domain("foundation/books/rust", "FOUNDATION")
    dm.create_domain("foundation/papers", "FOUNDATION")
    dm.create_domain("projects", "PROJECT")
    dm.create_domain("projects/luna_nine", "PROJECT")
    return dm
