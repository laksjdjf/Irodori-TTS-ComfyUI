"""
Resolve and import the irodori_tts package.

Search order:
1. Already importable (pip-installed or previously added to sys.path)
2. IRODORI_TTS_PATH environment variable
3. ~/Irodori-TTS  (default install location for the user)
4. Adjacent to the custom_nodes directory
"""
from __future__ import annotations

import os
import sys


def ensure_irodori_importable() -> None:
    try:
        import irodori_tts  # noqa: F401
        return
    except ImportError:
        pass

    candidates = []

    env_path = os.environ.get("IRODORI_TTS_PATH", "").strip()
    if env_path:
        candidates.append(env_path)

    # ~/Irodori-TTS
    candidates.append(os.path.expanduser("~/Irodori-TTS"))

    # <custom_nodes_dir>/Irodori-TTS  (sibling of this package)
    _self = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(os.path.dirname(_self), "Irodori-TTS"))

    last_error: ImportError | None = None
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, "irodori_tts")):
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                import irodori_tts  # noqa: F401
                return
            except ImportError as e:
                # e.g. a missing dependency inside irodori_tts, not a wrong path
                last_error = e
                continue

    raise ImportError(
        "irodori_tts package not found.\n"
        "Set the IRODORI_TTS_PATH environment variable to the Irodori-TTS repo root,\n"
        "or place the repo at ~/Irodori-TTS."
    ) from last_error
