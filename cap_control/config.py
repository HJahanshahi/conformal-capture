import os
"""
Configuration for cap_control.

All parameters in one place. Scripts import `from cap_control import config as cfg`
and override per-experiment.
"""
import numpy as np

# ============================================================
# Arm (matches capture_lib + space-robot-dq SRS arm)
# ============================================================
N_JOINTS = 7
ARM_REACH = 1.178                    # meters
JOINT_LIMITS_LOWER = np.array([-np.pi] * 7)
JOINT_LIMITS_UPPER = np.array([ np.pi] * 7)
JOINT_VEL_LIMIT    = 2.0             # rad/s, per joint
JOINT_TORQUE_LIMIT = 50.0            # N*m, per joint

# ============================================================
# Target (matches capture_lib)
# ============================================================
TARGET_STATE_DIM = 13                # [p(3), v(3), q(4), w(3)]
TARGET_OBS_DIM   = 7                 # [p(3), q(4)]
TARGET_OBS_INDICES = [0, 1, 2, 6, 7, 8, 9]

# Workspace placement (consistent with capture_lib feasibility analysis)
TARGET_DISTANCE_RATIO = 0.65         # initial position = 0.65 * ARM_REACH along +Z

# ============================================================
# UPN predictor
# ============================================================
UPN_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "capture_lib_v2", "trained_models_target_v3", "upn_target_v3.pt")
UPN_HIDDEN_DIM = 64
UPN_HISTORY_LEN = 10
UPN_UPDATE_FREQ = 5                  # Kalman update every N integration steps

# ============================================================
# Simulation
# ============================================================
SIM_DT = 0.01                        # seconds, inner simulator step
CONTROL_DT = 0.1                     # seconds, MPC re-solve interval
SENSOR_DT = 0.1                      # seconds, observation arrival interval
SIM_T_FINAL = 10.0                   # seconds, max simulation time

# Sensor noise (matches capture_lib training setup)
POS_NOISE_STD = 0.01                 # meters
ROT_NOISE_STD_DEG = 1.0              # degrees

# ============================================================
# MPC
# ============================================================
MPC_HORIZON_STEPS = 20               # decision steps (horizon = HORIZON_STEPS * CONTROL_DT)
MPC_SOLVER = "scipy"                 # "scipy" for Steps 4-6, "casadi" for Step 7+

# Cost weights (will be tuned per experiment)
W_TERMINAL_POS     = 100.0           # end-effector position error at T_grasp
W_TERMINAL_ORI     =  50.0           # end-effector orientation error at T_grasp
W_TERMINAL_VEL     =  10.0           # end-effector twist matching at T_grasp
W_RUNNING_TORQUE   =   0.01          # effort
W_RUNNING_SMOOTH   =   0.1           # joint-velocity smoothness
W_MANIP_MARGIN     =   5.0           # soft cost on low manipulability
MANIP_MIN          =   0.01          # matches capture_lib

# Use UPN covariance for Mahalanobis terminal cost (True for Step 7+)
USE_UNCERTAINTY_WEIGHTED_COST = False

# ============================================================
# Grasp trigger (reused from capture_lib conventions)
# ============================================================
GRASP_POS_TOL      = 0.05            # meters
GRASP_ORI_TOL_DEG  = 15.0            # degrees
GRASP_CONFIDENCE   = 0.90            # threshold

# ============================================================
# Logging / results
# ============================================================
RESULTS_DIR = "results"
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "capture_lib_v2", "trained_models_target_v3")
