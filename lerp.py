import torch

def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return torch.clamp(torch.linalg.norm(x, dim=dim, keepdim=True), min=eps)

@torch.no_grad()
def renormalized_lerp_retime(
    z0: torch.Tensor,
    zt: torch.Tensor,
    t,
    t_new,
    eps: float = 1e-8,
) -> torch.Tensor:
    # 1. Normalize inputs
    z0 = z0 / _safe_norm(z0, eps=eps)
    zt = zt / _safe_norm(zt, eps=eps)
    
    t = torch.as_tensor(t, dtype=z0.dtype, device=z0.device)
    t_new = torch.as_tensor(t_new, dtype=z0.dtype, device=z0.device)

    # 2. Get the dot product (cos similarity) between z0 and zt
    dot_0t = (z0 * zt).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    
    # 3. Solve for the magnitude R of the unnormalized vector (1-t)z0 + t*z1
    # Derived from: ||(1-t)z0 + t*z1||^2 = R^2
    # R satisfies: R^2 - 2(1-t)dot_0t*R + (1-2t) = 0
    # We use the quadratic formula: R = [ -b + sqrt(b^2 - 4ac) ] / 2
    b = -2.0 * (1.0 - t) * dot_0t
    c = 1.0 - 2.0 * t
    discriminant = torch.clamp(b**2 - 4.0 * c, min=0.0)
    
    # We take the larger root (+) because zt must point in the same direction as the interpolation
    R = (-b + torch.sqrt(discriminant)) / 2.0

    # 4. Express z_t_new as a linear combination of z0 and zt
    # Logic: (1-t_new)z0 + t_new*z1  can be rewritten using zt
    # Since z1 = (R*zt - (1-t)z0) / t
    # z_t_new_unnorm = (1 - t_new/t)z0 + (t_new * R / t) * zt
    # To avoid division by t, we multiply the whole vector by t (normalization handles it)
    
    weight_z0 = t * (1.0 - t_new) - t_new * (1.0 - t)
    weight_zt = t_new * R
    
    zt_new = weight_z0 * z0 + weight_zt * zt
    
    # 5. Final normalization
    return zt_new / _safe_norm(zt_new, eps=eps)

# ------------------- Reliability Test -------------------
if __name__ == "__main__":
    D = 2  # Hardest case for precision
    z0 = torch.randn(100, D)
    z1 = torch.randn(100, D)
    z0 /= z0.norm(dim=-1, keepdim=True)
    z1 /= z1.norm(dim=-1, keepdim=True)

    t, t_new = 0.3, 0.7

    def normed_lerp(a, b, t):
        x = (1 - t) * a + t * b
        return x / x.norm(dim=-1, keepdim=True)

    zt = normed_lerp(z0, z1, t)
    zt_true = normed_lerp(z0, z1, t_new)
    zt_hat = renormalized_lerp_retime(z0, zt, t, t_new)

    sim = torch.cosine_similarity(zt_true, zt_hat)
    print(f"Min Similarity: {sim.min().item():.8f}")
    print(f"Mean Similarity: {sim.mean().item():.8f}")