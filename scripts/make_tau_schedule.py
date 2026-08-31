"""Build path2_tau_schedule.py: the lookahead-schedule ablation. Identical controller and
mechanisms, but the decision signal is a PURE LOOKAHEAD SCHEDULE with matched
thresholds instead of the calibrated bound:

    q_surr(tau) = 40 deg * tau / 1.113 s

so rejection triggers at the same lookahead (tau ~ 1.11 s <-> 40 deg) and the
gain band spans the same lookaheads (0.556-1.113 s <-> 20-40 deg) as the
deployed conformal system. Any performance difference then isolates what the
calibrated SHAPE of q_hat(tau) contributes beyond a linear tau ramp; near-
equivalence supports the argument that conformal's contribution is knowing
WHERE the thresholds sit and keeping them calibrated under distribution change.

Run:  python make_tau_schedule.py    then:  python path2_tau_schedule.py
Output: path2_tau_schedule_results.json  (~15 min)
"""
import py_compile

SRC = "path2_conformal_v3.py"
DST = "path2_tau_schedule.py"

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


# tau at which the deployed calibrated bound crosses 40 deg (from the v2
# repeated-subsample table: 34.4 @ 1.0 s, 60.7 @ 1.5 s -> 1.113 s)
rep("surrogate ori bound",
    '''def q_hat_ori(lookahead):
    """Linearly interpolated calibrated orientation bound."""''',
    '''def q_hat_ori(lookahead):
    """ABLATION: pure lookahead schedule with matched thresholds
    (q_surr = 40 deg * tau / 1.113 s); calibrated table below is bypassed."""
    return float(40.0 * lookahead / 1.113)''')

rep("banner",
    'print(f"Stage 2+3 v2: COMBINED-RERUN conformal benchmark (45 traj x 3 seeds)")',
    'print(f"ABLATION: tau-schedule decision signal (45 traj x 3 seeds)")')

rep("output file",
    'with open("path2_conformal_v3_results.json", "w") as f:',
    'with open("path2_tau_schedule_results.json", "w") as f:')
rep("save message",
    'print("\\nRaw saved to path2_conformal_v3_results.json")',
    'print("\\nRaw saved to path2_tau_schedule_results.json")')

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
