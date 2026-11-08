"""Claim files: how concurrent jobs on a shared filesystem divide a queue of work between them.

A claim is a small JSON file named after the unit of work (a tuning block, a (problem, algorithm)
pair). Taking one is atomic: the claim file is CREATED with O_CREAT | O_EXCL, which the
filesystem's server decides, under a lock file that serialises the stale-claim takeover. Its
modification time is a heartbeat: the holder touches it periodically, and a claim whose mtime has
frozen for CLAIM_STALE_SECONDS belonged to a job that is gone and may be taken over. That is what
lets a killed job's work be picked up without anyone cleaning up after it.

Why creation must be exclusive: on a network filesystem every client caches directory lookups for
a while, so `os.path.exists(claim)` can answer "no" for up to a minute after another node created
the file. Deciding on that answer and then writing the claim can let two jobs hold the same
pair. An exclusive create is answered by the server and cannot be fooled by a stale cache; `beat()` and
`release()` touch the file only while its content still names this holder.

Used by the campaign runner (`experiment.running.queue`, one claim per result file, beaten from its
dispatch loop).
"""

import json
import os
import socket
import time
from contextlib import contextmanager

CLAIM_HEARTBEAT_SECONDS = 60
CLAIM_STALE_SECONDS = 600            # ten missed heartbeats: the claiming job is dead
# A directory another node has just created can stay invisible to this node's lookup cache for
# up to about a minute: an exclusive create inside it then fails with ENOENT although the directory
# exists. Creates retry that long.
NEW_DIRECTORY_GRACE_SECONDS = 60.0
NEW_DIRECTORY_RETRY_INTERVAL = 0.5


def _open_exclusively(path: str) -> int:
    """os.open(path, O_CREAT | O_EXCL | O_WRONLY), retrying ENOENT while the parent directory may
    still be invisible here (NEW_DIRECTORY_GRACE_SECONDS); FileExistsError passes through."""
    deadline = time.monotonic() + NEW_DIRECTORY_GRACE_SECONDS
    while True:
        try:
            return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            time.sleep(NEW_DIRECTORY_RETRY_INTERVAL)


@contextmanager
def file_lock(path: str):
    """Short-lived exclusive lock (an O_CREAT|O_EXCL lock file next to `path`) serialising
    read-merge-write cycles between concurrent jobs on a shared filesystem."""
    lock_path = path + ".lock"
    for _ in range(600):                 # wait up to ~60 s for the lock
        try:
            lock_fd = _open_exclusively(lock_path)
            break
        except FileExistsError:
            time.sleep(0.1)
    else:
        raise TimeoutError(f"Could not acquire {lock_path}. If a previous job crashed while "
                           f"holding it, delete the stale lock file and rerun.")
    try:
        yield
    finally:
        os.close(lock_fd)
        os.remove(lock_path)


def _holder():
    return {"job": os.environ.get("SLURM_JOB_ID", "local"), "host": socket.gethostname(), "pid": os.getpid()}


class Claim:
    """An "in progress" marker for one unit of work; `beat()` keeps it alive, `release()` ends it.
    Both act only while the file still names this holder: after a stale takeover the file belongs
    to another job, and a late beat or release must not keep alive or delete that job's claim."""

    def __init__(self, path: str, holder: dict | None = None):
        self.path = path
        self.holder = holder or _holder()

    def _is_mine(self) -> bool:
        try:
            with open(self.path) as f:
                stored = json.load(f)
        except (OSError, ValueError):
            return False
        return all(stored.get(k) == v for k, v in self.holder.items())

    def beat(self):
        try:
            if self._is_mine():
                os.utime(self.path, None)
        except OSError:
            pass                         # claim gone or filesystem hiccup; staleness handling covers it

    def release(self):
        try:
            if self._is_mine():
                os.remove(self.path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def is_held(claim_dir: str, name: str) -> bool:
    """Whether a live (not stale) claim on `name` exists - a cheap check that takes no lock, so a
    job can leave alone the files of a unit another job is working on without even opening them."""
    try:
        return time.time() - os.path.getmtime(os.path.join(claim_dir, f"{name}.claim")) < CLAIM_STALE_SECONDS
    except OSError:
        return False


def _create_exclusively(path: str, payload: str) -> bool:
    """Create `path` with `payload` if and only if it does not exist, decided by the filesystem's
    server (O_CREAT | O_EXCL); False when it exists."""
    try:
        fd = _open_exclusively(path)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(payload)
    return True


def try_claim(claim_dir: str, name: str, lock_path: str, finished=None):
    """Atomically claim the unit of work `name`. Returns a live Claim, or None when another job
    holds a fresh claim on it or `finished()` reports it done meanwhile. Stale claims are taken
    over with a message.

    The claim is created exclusively (see the module docstring), never decided by a lookup that a
    client cache may answer wrongly; the lock only serialises the takeover of a stale claim."""
    os.makedirs(claim_dir, exist_ok=True)
    claim_path = os.path.join(claim_dir, f"{name}.claim")
    holder = _holder()
    payload = json.dumps({**holder, "claimed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=4)
    with file_lock(lock_path):
        if finished is not None and finished():
            return None
        if _create_exclusively(claim_path, payload):
            return Claim(claim_path, holder)
        try:
            age = time.time() - os.stat(claim_path).st_mtime
        except FileNotFoundError:            # released between the create and the stat: try once more
            return Claim(claim_path, holder) if _create_exclusively(claim_path, payload) else None
        if age < CLAIM_STALE_SECONDS:
            return None
        print(f"Taking over stale claim for {name} (no heartbeat for {age:.0f} s).")
        try:
            os.remove(claim_path)
        except FileNotFoundError:
            pass
        return Claim(claim_path, holder) if _create_exclusively(claim_path, payload) else None
