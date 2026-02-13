"""Google Drive sync via rclone.

Periodically copies local data directories to Google Drive using the
pre-configured ``gdrive`` rclone remote.  Runs as a background async
task inside LiveEngine.

Requires:
    - rclone installed (~/bin/rclone or system PATH)
    - ``gdrive`` remote configured via ``rclone config``
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from quantlaxmi.data._paths import _PROJECT_ROOT, DATA_ROOT, EVENTS_DIR

logger = logging.getLogger(__name__)

# Locate rclone binary
_RCLONE_PATHS = [
    Path.home() / "bin" / "rclone",
    Path("/usr/local/bin/rclone"),
    Path("/usr/bin/rclone"),
]

LIVE_BARS_DIR = DATA_ROOT / "live_bars"
STATE_DIR = _PROJECT_ROOT / "data" / "state"


def _find_rclone() -> str | None:
    """Find rclone binary on disk."""
    for p in _RCLONE_PATHS:
        if p.exists():
            return str(p)
    # Fallback: check PATH
    found = shutil.which("rclone")
    return found


class GDriveSync:
    """Background rclone sync to Google Drive.

    Parameters
    ----------
    interval_sec : int
        Seconds between sync runs (default: 600 = 10 minutes).
    remote : str
        rclone remote name (default: ``gdrive``).
    """

    SYNC_DIRS: list[tuple[str, str]] = [
        (str(LIVE_BARS_DIR), "gdrive:QuantLaxmi/live_bars/"),
        (str(EVENTS_DIR), "gdrive:QuantLaxmi/events/"),
        (str(STATE_DIR), "gdrive:QuantLaxmi/state/"),
    ]

    def __init__(
        self,
        interval_sec: int = 600,
        remote: str = "gdrive",
    ):
        self.interval_sec = interval_sec
        self.remote = remote
        self._rclone = _find_rclone()
        self._task: asyncio.Task | None = None
        self._running = False
        self._sync_count = 0
        self._last_sync_error: str = ""

        if not self._rclone:
            logger.warning(
                "rclone not found; GDrive sync disabled. "
                "Install: curl https://rclone.org/install.sh | sudo bash"
            )

    async def start(self) -> None:
        """Start the periodic sync loop."""
        if not self._rclone:
            return
        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info(
            "GDriveSync started (interval=%ds, remote=%s)",
            self.interval_sec, self.remote,
        )

    async def stop(self) -> None:
        """Stop the sync loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _sync_loop(self) -> None:
        """Periodic sync loop."""
        try:
            while self._running:
                await asyncio.sleep(self.interval_sec)
                await self.sync_once()
        except asyncio.CancelledError:
            pass

    async def sync_once(self) -> bool:
        """Run a single sync pass (all directories)."""
        if not self._rclone:
            return False

        success = True
        for local, remote in self.SYNC_DIRS:
            local_path = Path(local)
            if not local_path.exists():
                continue

            # Replace remote name if configured differently
            actual_remote = remote.replace("gdrive:", f"{self.remote}:")

            try:
                proc = await asyncio.create_subprocess_exec(
                    self._rclone, "copy", local, actual_remote,
                    "--transfers", "4",
                    "--checkers", "8",
                    "--quiet",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=300,
                )

                if proc.returncode == 0:
                    logger.debug("Synced %s → %s", local, actual_remote)
                else:
                    err = stderr_bytes.decode(errors="replace") if stderr_bytes else f"exit code {proc.returncode}"  # noqa: E501
                    logger.warning("Sync failed %s → %s: %s", local, actual_remote, err)
                    self._last_sync_error = err
                    success = False
            except asyncio.TimeoutError:
                logger.warning("Sync timeout for %s", local)
                self._last_sync_error = "timeout"
                success = False
            except Exception as exc:
                logger.warning("Sync error for %s: %s", local, exc)
                self._last_sync_error = str(exc)
                success = False

        if success:
            self._sync_count += 1
            logger.info("GDrive sync #%d completed", self._sync_count)

        return success

    def stats(self) -> dict:
        """Return sync diagnostics."""
        return {
            "running": self._running,
            "rclone_found": self._rclone is not None,
            "sync_count": self._sync_count,
            "last_error": self._last_sync_error,
            "interval_sec": self.interval_sec,
            "remote": self.remote,
        }
