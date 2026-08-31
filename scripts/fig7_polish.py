"""Polish fig7 (3D trajectories): per-panel cameras, z-label collision fix,
distinct tick labels, dashed target path, time dots, solid markers.
Run once from the figures folder:  python fig7_polish.py
Then:  python make_all_figures.py
If a camera angle still reads badly, tweak the two numbers in VIEWS below
(elev, azim) per panel and rerun.
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

# 1. per-panel cameras + box aspect + z-label orientation
rep("cameras",
    "        ax.view_init(elev=25, azim=40)",
    "        elev, azim = [(22, -60), (20, -130)][idx]   # VIEWS: tweak here\n"
    "        ax.view_init(elev=elev, azim=azim)\n"
    "        ax.set_box_aspect((1, 1, 0.85))\n"
    "        ax.zaxis.set_rotate_label(False)")

# 2. z-label: vertical, modest pad (collision fix works with azim<0 -> z on left)
rep("zlabel",
    '        ax.set_zlabel("$z$ (m)", fontsize=10, labelpad=12)',
    '        ax.set_zlabel("$z$ (m)", fontsize=10, labelpad=6, rotation=90)')

# 3. distinct tick labels: 2 decimals, 3 bins
rep("tick bins",
    "            axis.set_major_locator(MaxNLocator(4))",
    "            axis.set_major_locator(MaxNLocator(3))")
rep("tick format",
    '            axis.set_major_formatter(FormatStrFormatter("%.1f"))',
    '            axis.set_major_formatter(FormatStrFormatter("%.2f"))')

# 4. gap annotation: per-panel offsets, clear of axes
rep("gap offsets",
    '                    xytext=(14, 14), textcoords="offset points",',
    '                    xytext=[(16, 14), (12, 18)][idx],\n'
    '                    textcoords="offset points",')

# 5. target path dashed (distinguish from EE), EE time dots each 1 s
rep("target dashed",
    '        ax.plot(grasp[:, 0], grasp[:, 1], grasp[:, 2], color=COLOR_GRASP,\n'
    '                linewidth=2.0, alpha=0.9, solid_capstyle="round",\n'
    '                label="Grasp point (target)")',
    '        ax.plot(grasp[:, 0], grasp[:, 1], grasp[:, 2], color=COLOR_GRASP,\n'
    '                linewidth=1.8, alpha=0.9, linestyle="--",\n'
    '                label="Grasp point (target)")\n'
    '        ax.scatter(ee[::10, 0], ee[::10, 1], ee[::10, 2],\n'
    '                    color=COLOR_EE, s=9, alpha=0.55, depthshade=False)')

# 6. solid marker colors (no depth fading)
for lab in ("EE start", "EE end", "Capture point"):
    rep(f"depthshade {lab}",
        f'label="{lab}", zorder=5)',
        f'label="{lab}", zorder=5, depthshade=False)')

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
