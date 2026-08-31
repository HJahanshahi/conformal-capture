"""fig11 (paper Fig 10) polish: show the 87.6-deg initial bound (was clipped
at ylim 80), anchor the 'drops below 20 deg' annotation at the true full-gain
step, and place each panel's legend to its right.
Run: python fig11_polish.py ; then: python make_all_figures.py
"""
import py_compile

PATH = "make_all_figures.py"
src = open(PATH, encoding="utf-8").read()
applied, missed = [], []

def rep(name, old, new, count=1):
    global src
    c = src.count(old)
    if c != count:
        missed.append(f"{name} (found {c}, expected {count})"); return
    src = src.replace(old, new); applied.append(name)

rep("red zone to 95",
    '    ax.fill_between(t, 40, 80, color="red", alpha=0.07)',
    '    ax.fill_between(t, 40, 95, color="red", alpha=0.07)')
rep("ylim 95",
    '    ax.set_ylabel("$\\\\hat{q}_{\\\\mathrm{ori}}$ (deg)", fontsize=10)\n'
    "    ax.set_ylim(0, 80)",
    '    ax.set_ylabel("$\\\\hat{q}_{\\\\mathrm{ori}}$ (deg)", fontsize=10)\n'
    "    ax.set_ylim(0, 95)")
rep("annotation anchor at full gain",
    '    gain_transition_idx = int(np.argmax(log["kp_ori_active"] > 2))',
    '    gain_transition_idx = int(np.argmax(np.asarray(log["kp_ori_active"]) > 4.9))')
rep("per-panel right legends",
    '''    handles, labels = [], []
    for src_ax in axes:
        h, l = src_ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    fig_legend_below(fig, handles, labels, ncol=1, fontsize=8)
    plt.tight_layout(h_pad=1.5)''',
    '''    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=8, frameon=True)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=8, frameon=True)
    plt.tight_layout(h_pad=1.5)''')

open(PATH, "w", encoding="utf-8").write(src)
print(f"APPLIED ({len(applied)}):")
for a in applied: print("  +", a)
if missed:
    print(f"MISSED ({len(missed)})")
    for m in missed: print("  !", m)
try:
    py_compile.compile(PATH, doraise=True); print("Compile check OK")
except py_compile.PyCompileError as e:
    print("COMPILE FAILED:", e)
