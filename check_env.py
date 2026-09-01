r"""Diagnose the space_robot_dq environment mismatch.

Run twice and compare:
    python check_env.py
    <path-to-your-venv>\Scripts\python.exe check_env.py
"""
import sys, os

print("interpreter :", sys.executable)
print("cwd         :", os.getcwd())

try:
    import space_robot_dq as srd
    print("space_robot_dq:", getattr(srd, "__version__", "(no __version__)"),
          "\n  at        :", os.path.dirname(srd.__file__))
except Exception as e:
    print("space_robot_dq: NOT IMPORTABLE ->", e)

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cap_control.dynamics.free_floating import FreeFloatingChaser
    ch = FreeFloatingChaser()
    dyn = ch.dyn
    print("dynamics cls:", type(dyn).__name__, "from", type(dyn).__module__)
    print("  module file:", sys.modules[type(dyn).__module__].__file__)
    needed = ["compute_effective_arm_inertia", "compute_coriolis_term",
              "compute_generalized_jacobian", "compute_base_velocity"]
    for m in needed:
        print(f"  {m:34s}", "OK" if hasattr(dyn, m) else "MISSING")
    inertia_like = [m for m in dir(dyn) if "inertia" in m.lower()]
    print("  inertia-related methods present:", inertia_like)
except Exception as e:
    print("chaser construction failed ->", type(e).__name__, e)
