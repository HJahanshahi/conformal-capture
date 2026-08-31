"""fig7 polish v2: restore (a) camera, mirror (b) to z-left family, per-side
gap labels, wider panel gap. Run: python fig7_polish2.py ; then regenerate."""
import py_compile
PATH = "make_all_figures.py"
src = open(PATH, encoding="utf-8").read()
E = [('        elev, azim = [(22, -60), (20, -130)][idx]   # VIEWS: tweak here', '        elev, azim = [(25, 40), (20, 50)][idx]   # VIEWS: tweak here'), ('                    xytext=[(16, 14), (12, 18)][idx],\n                    textcoords="offset points",\n                    fontsize=9, ha="left", color="black", fontweight="bold",', '                    xytext=[(-12, 16), (12, 18)][idx],\n                    textcoords="offset points",\n                    fontsize=9, ha=["right", "left"][idx],\n                    color="black", fontweight="bold",'), ('    plt.tight_layout(rect=[0, 0.07, 1, 0.97], w_pad=2)', '    plt.tight_layout(rect=[0, 0.07, 1, 0.97], w_pad=3.5)')]
ok = 0
for old, new in E:
    if src.count(old) == 1:
        src = src.replace(old, new); ok += 1
    else:
        print("MISSED:", old[:70])
open(PATH, "w", encoding="utf-8").write(src)
print(f"applied {ok}/{len(E)}")
py_compile.compile(PATH, doraise=True); print("Compile check OK")
