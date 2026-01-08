#!/usr/bin/env python3
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "ggml-org/llama.cpp"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def compute_checksum(asset_path: Path) -> str:
    result = subprocess.run(
        ["swift", "package", "compute-checksum", str(asset_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    package_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Package.swift")
    if not package_file.is_file():
        print(f"error: {package_file} not found", file=sys.stderr)
        return 1
    resolved_file = package_file.parent / "Package.resolved"

    repo_data = get_json(f"https://api.github.com/repos/{REPO}")
    default_branch = repo_data["default_branch"]
    latest_sha = get_json(
        f"https://api.github.com/repos/{REPO}/commits/{default_branch}"
    )["sha"]
    release = get_json(f"https://api.github.com/repos/{REPO}/releases/latest")

    asset_url = None
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if name.startswith("llama-") and name.endswith("xcframework.zip"):
            asset_url = asset.get("browser_download_url")
            break

    if not asset_url:
        print("error: xcframework asset not found in latest release", file=sys.stderr)
        return 1

    asset_name = asset_url.rsplit("/", 1)[-1]
    asset_path = package_file.parent / asset_name
    try:
        with urllib.request.urlopen(asset_url) as resp, asset_path.open("wb") as f:
            f.write(resp.read())
        checksum = compute_checksum(asset_path)
    finally:
        if asset_path.exists():
            asset_path.unlink()

    text = package_file.read_text()
    updated = text
    updated = re.sub(
        r'(\.package\(url: "https://github.com/ggml-org/llama.cpp.git", revision: ")([0-9a-f]+)(")',
        r"\g<1>" + latest_sha + r"\g<3>",
        updated,
    )
    updated = re.sub(
        r'(url: ")(https://github.com/ggml-org/llama.cpp/releases/download/[^"]+)(")',
        r"\g<1>" + asset_url + r"\g<3>",
        updated,
    )
    updated = re.sub(
        r'(checksum: ")([0-9a-f]+)(")',
        r"\g<1>" + checksum + r"\g<3>",
        updated,
    )

    if updated == text:
        print(f"{package_file} already up to date")
        print(f"- revision ({default_branch}): {latest_sha}")
        print(f"- LlamaFramework url: {asset_url}")
        print(f"- LlamaFramework checksum: {checksum}")
    else:
        package_file.write_text(updated)
        print(f"Updated {package_file}")
        print(f"- revision ({default_branch}): {latest_sha}")
        print(f"- LlamaFramework url: {asset_url}")
        print(f"- LlamaFramework checksum: {checksum}")

    if resolved_file.is_file():
        resolved_text = resolved_file.read_text()
        resolved = json.loads(resolved_text)
        updated_resolved = False
        for pin in resolved.get("pins", []):
            if pin.get("identity") == "llama.cpp":
                state = pin.get("state", {})
                if state.get("branch") != default_branch:
                    state["branch"] = default_branch
                    updated_resolved = True
                if state.get("revision") != latest_sha:
                    state["revision"] = latest_sha
                    updated_resolved = True
                pin["state"] = state
        if updated_resolved:
            resolved_file.write_text(
                json.dumps(resolved, indent=2, separators=(",", " : ")) + "\n"
            )
            print(f"Updated {resolved_file}")
        else:
            print(f"{resolved_file} already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
