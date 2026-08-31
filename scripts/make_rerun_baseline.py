"""Build path2_winning_v2.py: baseline benchmark with ALIGNED evaluation.

Only two changes vs path2_full_winning.py (original untouched):
  1. Errors compare the post-step chaser state with the target at t_now + dt
     (same evaluation convention as path2_conformal_v3.py).
  2. Output -> path2_winning_v2_results.json

Run:  python make_rerun_baseline.py   then:  python path2_winning_v2.py
"""
import py_compile

SRC = r"phase1_final\path2_full_winning.py"
DST = "path2_winning_v2.py"

src = open(SRC, encoding="utf-8").read()
applied, missed = [], []


def rep(name, old, new, count=1):
    global src
    c = src.count(old)
    if c != count:
        missed.append(f"{name} (found {c}, expected {count})")
        return
    src = src.replace(old, new)
    applied.append(name)


rep("aligned evaluation",
    "        true_st = target_sim.state_at(t_now)",
    "        true_st = target_sim.state_at(t_now + dt)"
    "  # aligned with post-step chaser state")
rep("output filename",
    'with open("path2_winning_results.json", "w") as f:',
    'with open("path2_winning_v2_results.json", "w") as f:')
rep("save message",
    'print("\\nRaw saved to path2_winning_results.json")',
    'print("\\nRaw saved to path2_winning_v2_results.json")')

open(DST, "w", encoding="utf-8").write(src)
print(f"APPLIED ({len(applied)}):")
for a in applied:
    print("  +", a)
if missed:
    print(f"MISSED ({len(missed)})")
    for m in missed:
        print("  !", m)
try:
    py_compile.compile(DST, doraise=True)
    print(f"Compile check OK -> {DST}")
except py_compile.PyCompileError as e:
    print("COMPILE FAILED:", e)
