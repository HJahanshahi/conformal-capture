# Supporting studies

These scripts produce the selection and verification results reported in the
article. They are not part of the main reproduction pipeline; run them
individually from the repository root.

* `sweep_gains.py` - gain and torque-limit selection on the validation split (Section 7)
* `select_design.py` - mechanism component study reported in Table 5 (Section 7.9)
* `sweep_mechanisms.py` - first-pass component study
* `convergence_test.py` - plant integration convergence study (Section 7)
* `check_coriolis.py` - passivity check of the Coriolis construction (Section 5.4)
