"""Full local snapshots. Only load trusted snapshots produced by this project.

Pickle retains types and shared references across memories, scratch and maze.
It is intentionally not an import format for downloaded/untrusted artifacts.
"""
import gzip
import hashlib
import json
import os
import pickle
import random
from datetime import date, datetime
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def rng_state():
    import numpy as np
    return {"python": random.getstate(), "numpy": np.random.get_state()}


def restore_rng(state):
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def _canonical(value):
    import numpy as np
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return {"datetime": value.isoformat()}
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        pairs = [(_canonical(k), _canonical(v)) for k, v in value.items()]
        return {"dict": sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
    if isinstance(value, set):
        return {"set": sorted((_canonical(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [_canonical(v) for v in value]}
    if hasattr(value, "__dict__"):
        return {"class": type(value).__module__ + "." + type(value).__name__,
                "state": _canonical(vars(value))}
    raise TypeError(f"Unhashable pilot state type: {type(value)}")


def state_digest(server, rng=None):
    data = _canonical({"server": server, "rng": rng if rng is not None else rng_state()})
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def save_snapshot(path, server, protocol_state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = rng_state()
    data = {"format": 1, "server": server, "rng": rng,
            "protocol": protocol_state, "state_digest": state_digest(server, rng)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as out:
            pickle.dump(data, out, protocol=4)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)
    return data["state_digest"]


def load_snapshot(path, restore_random=True):
    with gzip.open(path, "rb") as inp:
        data = pickle.load(inp)
    if data["format"] != 1 or state_digest(data["server"], data["rng"]) != data["state_digest"]:
        raise ValueError("Snapshot state validation failed")
    if restore_random:
        restore_rng(data["rng"])
    return data


def materialize_environment(server, folder):
    """Write the *next* perception input, not the previous tile positions."""
    folder = Path(folder)
    previous = folder / "movement" / f"{server.step - 1}.json"
    if server.step and previous.exists():
        movements = json.loads(previous.read_text())["persona"]
        positions = {name: {"x": info["movement"][0], "y": info["movement"][1]}
                     for name, info in movements.items()}
    elif hasattr(server, "pilot_next_environment"):
        positions = server.pilot_next_environment
    else:
        positions = {name: {"x": xy[0], "y": xy[1]} for name, xy in server.personas_tile.items()}
    server.pilot_next_environment = positions
    atomic_json(folder / "environment" / f"{server.step}.json", positions)
