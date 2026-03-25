from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from pymatgen.core.structure import Structure


def _load_mlip_calc():
    module_path = Path(__file__).resolve().with_name("mlip.py")
    spec = importlib.util.spec_from_file_location("interoptimus_mlip_standalone", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load MLIP module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MlipCalc


def main() -> None:
    payload = json.loads(sys.stdin.read())
    calc = payload["calc"]
    operation = payload["operation"]
    structure = Structure.from_dict(payload["structure"])
    user_settings = dict(payload.get("user_settings") or {})
    user_settings["_force_local"] = True
    optimizer = payload.get("optimizer", "BFGS")
    kwargs = dict(payload.get("kwargs") or {})

    MlipCalc = _load_mlip_calc()
    mlip = MlipCalc(calc=calc, user_settings=user_settings)
    if operation == "calculate":
        energy = mlip.calculate(structure)
        print(json.dumps({"energy": float(energy)}))
        return
    if operation == "optimize":
        relaxed, energy = mlip.optimize(structure, optimizer=optimizer, **kwargs)
        print(
            json.dumps(
                {
                    "structure": relaxed.as_dict(),
                    "energy": float(energy),
                }
            )
        )
        return
    raise ValueError(f"Unsupported external MLIP operation: {operation}")


if __name__ == "__main__":
    main()
