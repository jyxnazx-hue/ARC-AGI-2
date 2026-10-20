# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import numpy as np

def mask_ratio(tau: float) -> float:
    """
    Cosine schedule from the DRM paper (Equation 8).
    tau=0 -> 0% masked. tau=1 -> 100% masked.
    """
    alpha_bar = np.cos(tau * np.pi / 2) ** 2
    return 1.0 - alpha_bar

def corrupt_canvas(
    canvas: torch.Tensor, 
    cell_mask: torch.Tensor, 
    tau: float, 
    mask_token_id: int = 1,
    pad_token_id: int = 0
) -> torch.Tensor:
    
    r = mask_ratio(tau)
    corrupted = canvas.clone()
    
    valid_indices = torch.where(cell_mask == 1)[0]
    n_valid = len(valid_indices)
    if n_valid == 0:
        return corrupted
        
    n_mask = int(np.round(r * n_valid + 1e-6))
    
    if n_mask > 0:
        mask_indices = valid_indices[torch.randperm(n_valid)[:n_mask]]
        corrupted[mask_indices] = mask_token_id
        
    return corrupted