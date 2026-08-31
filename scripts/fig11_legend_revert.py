"""fig11 legend revert: restore the ORIGINAL single combined legend below the
whole figure (two-column friendly), keeping the ylim/zone/annotation fixes.
Handles either prior legend state (right-side or per-panel bottom).
Run: python fig11_legend_revert.py ; then: python make_all_figures.py
"""
import py_compile

PATH = "make_all_figures.py"
src = open(PATH, encoding="utf-8").read()

ORIGINAL = '''    handles, labels = [], []
    for src_ax in axes:
        h, l = src_ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    fig_legend_below(fig, handles, labels, ncol=1, fontsize=8)
    plt.tight_layout(h_pad=1.5)'''

VARIANT_RIGHT = '''    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=8, frameon=True)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=8, frameon=True)
    plt.tight_layout(h_pad=1.5)'''

VARIANT_BOTTOM = '''    legend_below(axes[0], ncol=1, yoff=-0.24, fontsize=8)
    legend_below(axes[1], ncol=1, yoff=-0.42, fontsize=8)
    plt.tight_layout(h_pad=7.0)'''

done = False
for variant, name in ((VARIANT_RIGHT, "right-side"), (VARIANT_BOTTOM, "per-panel bottom")):
    if src.count(variant) == 1:
        src = src.replace(variant, ORIGINAL)
        print(f"reverted {name} legends -> original combined legend")
        done = True
        break
if not done:
    if src.count(ORIGINAL) == 1:
        print("already at original combined legend - nothing to do")
    else:
        print("MISSED: no known legend block found")

open(PATH, "w", encoding="utf-8").write(src)
py_compile.compile(PATH, doraise=True)
print("Compile check OK")
