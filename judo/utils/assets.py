# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import os
import time
import zipfile
from pathlib import Path

import requests

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def mesh_downloads_disabled() -> bool:
    """Return whether judo should skip downloading mesh assets from GitHub releases.

    Network downloads are enabled by default. Third-party or offline deployments (e.g. air-gapped
    robots, hermetic build systems that vendor meshes themselves) can opt out by setting the
    ``JUDO_OFFLINE`` environment variable to a truthy value (``1``, ``true``, ``yes``, ``on``).
    When disabled, :func:`download_and_extract_meshes` becomes a no-op and callers are expected to
    provide the meshes through other means (e.g. a copy already present on disk).
    """
    return os.environ.get("JUDO_OFFLINE", "").strip().lower() in _TRUTHY_ENV_VALUES


def acquire_lock(lock_path: Path, timeout: int = 60, poll_interval: float = 0.1) -> None:
    """Acquire a lock by creating a lock file atomically.

    Raises TimeoutError if the lock can't be acquired in `timeout` seconds.
    """
    start = time.time()
    while True:
        try:
            # open with O_CREAT | O_EXCL to create file atomically, fail if exists
            os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            return  # acquired
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for lock file {lock_path}") from None
            time.sleep(poll_interval)


def release_lock(lock_path: Path) -> None:
    """Remove the lock file to release the lock."""
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def download_and_extract_meshes(
    extract_root: str,
    repo: str = "bdaiinstitute/judo",
    asset_name: str = "meshes.zip",
    tag: str | None = None,
) -> None:
    """Downloads meshes.zip from the latest public GitHub release and extracts it.

    This is a no-op when mesh downloads are disabled via the ``JUDO_OFFLINE`` environment variable
    (see :func:`mesh_downloads_disabled`), allowing offline/third-party deployments to supply their
    own meshes without any network access.
    """
    if mesh_downloads_disabled():
        return

    extract_path = Path(extract_root).expanduser()
    meshes_path = extract_path / "meshes"
    lock_path = extract_path / ".meshes_download.lock"

    try:
        acquire_lock(lock_path)  # prevent race conditions resulting in multiple downloads

        # case: meshes already extracted
        if meshes_path.exists():
            return

        # fetch latest release info
        print("Mesh assets not detected! Downloading assets now...")
        if tag is None:
            api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        else:
            api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        headers = {}
        gh_token = os.environ.get("GITHUB_TOKEN")
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        release_data = response.json()

        # get the download URL for meshes.zip
        asset_url = None
        for asset in release_data.get("assets", []):
            if asset["name"] == asset_name:
                asset_url = asset["browser_download_url"]
                break
        if asset_url is None:
            raise ValueError(f"{asset_name} not found in latest release of {repo}.")

        # download and extract
        zip_path = meshes_path.with_suffix(".zip")
        meshes_path.mkdir(parents=True, exist_ok=True)
        with requests.get(asset_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        if zip_path.exists():
            zip_path.unlink()  # remove the zip file after extraction

    finally:
        release_lock(lock_path)
