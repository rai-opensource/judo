# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import os
import time
import zipfile
from pathlib import Path

import requests


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


def _request_with_retry(
    method: str,
    url: str,
    *,
    retries: int,
    timeout: float,
    backoff_time: float,
    **request_kwargs,
) -> requests.Response:
    """Execute an HTTP request with retries."""
    last_exception: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.request(method, url, timeout=timeout, **request_kwargs)
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt >= retries - 1:
                raise
            last_exception = exc
            time.sleep(backoff_time)
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected retry loop termination in _request_with_retry")


def download_and_extract_meshes(
    extract_root: str,
    repo: str = "bdaiinstitute/judo",
    asset_name: str = "meshes.zip",
    tag: str | None = None,
    retries: int = 3,
    timeout: float = 30.0,
    backoff_time: float = 1.0,
) -> None:
    """Downloads meshes.zip from the latest public GitHub release and extracts it."""
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
        response = _request_with_retry(
            "GET",
            api_url,
            retries=retries,
            timeout=timeout,
            backoff_time=backoff_time,
            headers=headers,
        )
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
        with _request_with_retry(
            "GET",
            asset_url,
            retries=retries,
            timeout=timeout,
            backoff_time=backoff_time,
            stream=True,
        ) as r:
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
