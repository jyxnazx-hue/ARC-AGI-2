# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import numpy as np
from train_trm import Trainer, REAL, SMOKE, main as train_trm_main
from src.corrupt import corrupt_canvas
from src.core import dataset as ds
# pyrefly: ignore[missing-import]
import torch.nn.functional as F

class DRMTrainer(Trainer):
    def train_step(self) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum = max(1, self.cfg.grad_accum_steps)
        total_loss = 0.0

        for _ in range(accum):
            batch = self._next_batch()
            B = batch["output_canvases"].shape[0]
            
            tau = torch.rand(B, device=self.cfg.device)
            
            clean_outputs = batch["output_canvases"].clone()
            
            corrupted_outputs = batch["output_canvases"].clone()
            for i in range(B):
                corrupted_outputs[i, -1] = corrupt_canvas(
                    canvas=batch["output_canvases"][i, -1],
                    cell_mask=batch["output_cell_mask"][i, -1],
                    tau=tau[i].item()
                )
            batch["output_canvases"] = corrupted_outputs
            
            token_ids, attention, loss_mask = ds.build_sequence(batch, puzzle_len=self.cfg.puzzle_len)
            
            clean_batch = {**batch, "output_canvases": clean_outputs}
            clean_token_ids, _, _ = ds.build_sequence(clean_batch, puzzle_len=self.cfg.puzzle_len)
            
            token_ids = token_ids.to(self.cfg.device)
            attention = attention.to(self.cfg.device)
            loss_mask = loss_mask.to(self.cfg.device)
            clean_token_ids = clean_token_ids.to(self.cfg.device)
            puzzle_idx = batch["puzzle_idx"].to(self.cfg.device)

            out = self.model(
                token_ids, puzzle_idx,
                attention_mask=attention, loss_mask=loss_mask,
                tbptt_window=self.cfg.tbptt_window,
                tau=tau,
            )
            
            logits = out["final_logits"]
            
            masked_targets = torch.where(loss_mask, clean_token_ids, -100)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                masked_targets.view(-1),
                ignore_index=-100
            )

            (loss / accum).backward()
            total_loss += loss.item() / accum

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update(self.model)
        self.step += 1

        return {
            "loss": total_loss,
            "lr_model": self.optimizer.param_groups[0]["lr"],
            "lr_puzzle": self.optimizer.param_groups[1]["lr"],
        }

def main():
    import argparse
    from pathlib import Path
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["smoke", "real"], default="smoke")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = SMOKE if args.config == "smoke" else REAL
    if args.data_root: cfg.data_root = Path(args.data_root)
    if args.time_budget_min is not None: cfg.time_budget_min = args.time_budget_min
    if args.total_steps is not None: cfg.total_steps = args.total_steps
    if args.batch_size is not None: cfg.batch_size = args.batch_size

    trainer = DRMTrainer(cfg)
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    trainer.train()

if __name__ == "__main__":
    main()
