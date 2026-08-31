"""
Abstract MPC interface.

Concrete MPCs (ApproachMPC, GraspMPC, DetumbleMPC) implement `.solve(state, reference)`
and own their own cost function / constraints / solver.
"""
from abc import ABC, abstractmethod


class MPCBase(ABC):
    """Common interface so the simulation loop can swap controllers."""

    @abstractmethod
    def solve(self, state, reference):
        """
        Given the current state and a reference, return a command.

        Returns
        -------
        u0 : numpy array
            The FIRST control command to apply to the plant now.
            Subsequent commands in the optimized sequence are discarded
            (receding horizon -- we re-solve next cycle).
        info : dict
            Solver diagnostics (solve_time_s, cost, success, ...).
        """
        ...
