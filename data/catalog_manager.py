"""Catalog manager for tracking processed data artifacts."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


class DataCatalog:
    """Manages data catalog with checksums and metadata."""

    def __init__(self, catalog_path: str = "data/catalog.yaml"):
        self.catalog_path = Path(catalog_path)
        self.catalog = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load existing catalog or create empty one."""
        if self.catalog_path.exists():
            with open(self.catalog_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save(self):
        """Save catalog to disk."""
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, "w") as f:
            yaml.dump(self.catalog, f, default_flow_style=False, sort_keys=False)

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _count_rows(self, file_path: Path) -> int:
        """Count rows in file based on extension."""
        if file_path.suffix == ".parquet":
            import pandas as pd
            return len(pd.read_parquet(file_path))
        elif file_path.suffix == ".jsonl":
            with open(file_path) as f:
                return sum(1 for _ in f)
        elif file_path.suffix == ".csv":
            import pandas as pd
            return len(pd.read_csv(file_path))
        return 0

    def register(self, name: str, file_path: Path, metadata: Dict[str, Any] = None):
        """Register or update an artifact in the catalog."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        entry = {
            "path": str(file_path),
            "rows": self._count_rows(file_path),
            "checksum_sha256": self._compute_checksum(file_path),
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "size_bytes": file_path.stat().st_size,
        }

        if metadata:
            entry.update(metadata)

        self.catalog[name] = entry
        self._save()

    def get(self, name: str) -> Dict[str, Any]:
        """Get artifact metadata from catalog."""
        return self.catalog.get(name)

    def exists(self, name: str) -> bool:
        """Check if artifact exists in catalog."""
        return name in self.catalog

    def needs_update(self, name: str, file_path: Path) -> bool:
        """Check if file needs to be reprocessed."""
        if not self.exists(name):
            return True
        if not file_path.exists():
            return True

        entry = self.get(name)
        current_checksum = self._compute_checksum(file_path)
        return current_checksum != entry.get("checksum_sha256")

    def list_all(self) -> Dict[str, Any]:
        """Return all catalog entries."""
        return self.catalog

    def summary(self) -> str:
        """Return human-readable summary of catalog."""
        lines = ["Data Catalog Summary", "=" * 60]
        for name, entry in self.catalog.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Path: {entry['path']}")
            lines.append(f"  Rows: {entry['rows']:,}")
            lines.append(f"  Size: {entry['size_bytes'] / 1024 / 1024:.2f} MB")
            lines.append(f"  Updated: {entry['last_updated']}")
            if "stats" in entry:
                lines.append(f"  Stats: {json.dumps(entry['stats'])}")
        return "\n".join(lines)

