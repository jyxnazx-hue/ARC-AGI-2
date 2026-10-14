from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
# pyrefly: ignore [missing-import]
import numpy as np


class ARCTokenizer:
    def __init__(self):
        tokens = [
            "<pad>", "<bos>", "<eos>", "<unk>", "<sep>",
            "<start_example>", "<end_example>",
            "<start_input>", "<end_input>",
            "<start_output>", "<end_output>",
            "<start_row>", "<end_row>",
            "<row_by_row>", "<snake>",
            "<task_solve>", "<task_denoise>"
        ]
        tokens += [f"<c{i}>" for i in range(10)]
        tokens += [f"<puzzle_{i}>" for i in range(16)]

        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(tokens)}
        self.id_to_token: Dict[int, str] = {idx: tok for idx, tok in enumerate(tokens)}

        self.pad_token_id = self.token_to_id["<pad>"]
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]
        self.unk_token_id = self.token_to_id["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, "<unk>") for i in ids]

    def grid_to_tokens(self, grid: np.ndarray, mode: str = "row_by_row") -> List[str]:
        tokens = [f"<{mode}>"]
        h, w = grid.shape
        for r in range(h):
            tokens.append("<start_row>")
            cols = range(w) if (mode == "row_by_row" or r % 2 == 0) else range(w - 1, -1, -1)
            for c in cols:
                tokens.append(f"<c{int(grid[r, c])}>")
            tokens.append("<end_row>")
        return tokens

    def tokens_to_grid(self, tokens: List[str]) -> np.ndarray:
        mode = "row_by_row"
        if "<snake>" in tokens:
            mode = "snake"

        rows = []
        curr_row = []
        inside_row = False

        for t in tokens:
            if t == "<start_row>":
                curr_row = []
                inside_row = True
            elif t == "<end_row>":
                if inside_row:
                    rows.append(curr_row)
                    inside_row = False
            elif inside_row and t.startswith("<c") and t[2:-1].isdigit():
                curr_row.append(int(t[2:-1]))

        if not rows:
            return np.zeros((0, 0), dtype=np.int8)

        grid = np.zeros((len(rows), len(rows[0])), dtype=np.int8)
        for r_idx, r_data in enumerate(rows):
            if mode == "snake" and r_idx % 2 == 1:
                r_data = r_data[::-1]
            grid[r_idx, :] = r_data

        return grid

    def serialize_task(
        self,
        task: Dict[str, Any],
        test_idx: int = 0,
        mode: str = "row_by_row",
        include_target: bool = False
    ) -> Tuple[List[str], Optional[List[str]]]:
        tokens = ["<bos>", f"<{mode}>"]

        for inp, out in task["train"]:
            tokens.append("<start_example>")
            tokens.append("<start_input>")
            tokens.extend(self.grid_to_tokens(inp, mode=mode))
            tokens.append("<end_input>")
            tokens.append("<start_output>")
            tokens.extend(self.grid_to_tokens(out, mode=mode))
            tokens.append("<end_output>")
            tokens.append("<end_example>")

        test_inp, test_out = task["test"][test_idx]
        tokens.append("<start_example>")
        tokens.append("<start_input>")
        tokens.extend(self.grid_to_tokens(test_inp, mode=mode))
        tokens.append("<end_input>")
        tokens.append("<start_output>")

        target_tokens = None
        if include_target and test_out is not None:
            target_tokens = self.grid_to_tokens(test_out, mode=mode) + ["<end_output>", "<end_example>", "<eos>"]
        elif not include_target:
            tokens.append("<eos>")

        return tokens, target_tokens

    def deserialize_task_output(self, tokens: List[str]) -> np.ndarray:
        return self.tokens_to_grid(tokens)