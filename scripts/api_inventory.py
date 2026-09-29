"""T1.1 — API contract inventory.

Introspects every public callable in the engine and records its signature, so
call-site drift (the class of bug found 9+ times) can be caught. Writes
assets/api_inventory.json.

Not a test: an inventory used to spot methods whose signature was never verified
by a call site.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import pkgutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import super_solver  # noqa: E402


def main():
    inv = {}
    for modinfo in pkgutil.walk_packages(super_solver.__path__, prefix="super_solver."):
        name = modinfo.name
        if "__pycache__" in name:
            continue
        try:
            mod = importlib.import_module(name)
        except Exception as e:
            inv[name] = {"_import_error": f"{type(e).__name__}: {e}"}
            continue
        members = {}
        for mname, obj in vars(mod).items():
            if mname.startswith("_"):
                continue
            if inspect.isfunction(obj) or inspect.isclass(obj):
                # skip re-exported symbols from other modules
                if getattr(obj, "__module__", None) != name:
                    continue
                if inspect.isclass(obj):
                    methods = {}
                    for meth_name, meth in vars(obj).items():
                        if meth_name.startswith("_") or not callable(meth):
                            continue
                        try:
                            methods[meth_name] = str(inspect.signature(meth))
                        except (TypeError, ValueError):
                            methods[meth_name] = "?"
                    members[mname] = {"kind": "class", "methods": methods}
                else:
                    try:
                        members[mname] = {"kind": "function",
                                          "sig": str(inspect.signature(obj))}
                    except (TypeError, ValueError):
                        members[mname] = {"kind": "function", "sig": "?"}
        if members:
            inv[name] = members

    out = os.path.join(os.path.dirname(__file__), "..", "assets", "api_inventory.json")
    json.dump(inv, open(out, "w"), indent=2)
    n_classes = sum(
        1 for m in inv.values() if isinstance(m, dict)
        for v in m.values() if isinstance(v, dict) and v.get("kind") == "class"
    )
    n_funcs = sum(
        1 for m in inv.values() if isinstance(m, dict)
        for v in m.values() if isinstance(v, dict) and v.get("kind") == "function"
    )
    print(f"modules={sum(1 for m in inv.values() if isinstance(m, dict))} "
          f"classes={n_classes} functions={n_funcs}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
