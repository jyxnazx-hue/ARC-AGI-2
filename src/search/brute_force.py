"""
src/search/brute_force.py
Deterministic symbolic searcher evaluating 1-to-2 step DSL compositions on ARC tasks.
"""

import itertools
from typing import Dict, Any, List, Optional, Tuple
# pyrefly: ignore [missing-import]
import numpy as np
from src.core.grid import Grid
from src.core.guardrails import check_dimension_bounds, check_color_consistency
from src.dsl.interpreter import DSLInterpreter, PRIMITIVE_REGISTRY


class BruteForceSearcher:
    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth
        self.primitives = list(PRIMITIVE_REGISTRY.keys())

    def solve_task(self, task_data: Dict[str, Any]) -> List[List[List[int]]]:
        """
        Searches for a DSL sequence that yields 100% match on all train pairs.
        Returns a list containing up to 2 attempts for each test input (Pass@2 format).
        """
        train_pairs: List[Tuple[Grid, Grid]] = [
            (Grid(pair["input"]), Grid(pair["output"]))
            for pair in task_data.get("train", [])
        ]
        test_inputs: List[Grid] = [
            Grid(pair["input"]) for pair in task_data.get("test", [])
        ]

        solved_program: Optional[List[str]] = None

        # Search depths: 1-step, then 2-step compositions
        for depth in range(1, self.max_depth + 1):
            for candidate_prog in itertools.product(self.primitives, repeat=depth):
                prog = list(candidate_prog)
                if self._verify_on_train(prog, train_pairs):
                    solved_program = prog
                    break
            if solved_program is not None:
                break

        # Generate Pass@2 predictions for each test instance
        test_results = []
        for t_in in test_inputs:
            if solved_program is not None:
                try:
                    pred_grid = DSLInterpreter.execute(solved_program, t_in)
                    # Run guardrail verification
                    if check_dimension_bounds(pred_grid) and check_color_consistency(pred_grid, t_in, train_pairs):
                        attempt_1 = pred_grid.to_list()
                    else:
                        attempt_1 = t_in.to_list()
                except Exception:
                    attempt_1 = t_in.to_list()
            else:
                attempt_1 = t_in.to_list()

            # Attempt 2 fallback (identity/input canvas)
            attempt_2 = t_in.to_list()
            test_results.append({
                "attempt_1": attempt_1,
                "attempt_2": attempt_2
            })

        return test_results

    def _verify_on_train(self, program: List[str], train_pairs: List[Tuple[Grid, Grid]]) -> bool:
        """Checks if the candidate program achieves exact pixel match on all train pairs."""
        for g_in, g_out_target in train_pairs:
            try:
                g_pred = DSLInterpreter.execute(program, g_in)
                if g_pred != g_out_target:
                    return False
            except Exception:
                return False
        return True