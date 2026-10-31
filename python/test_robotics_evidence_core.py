"""Unit tests for robotics_evidence_core.py."""

import numpy as np
import torch

from robotics_evidence_core import (
    CrossModalSE3Fusion,
    TaiwanComplianceGuard,
    logit_adjusted_loss,
    preprocess_rgbd_lwir_opencv,
    verify_block_temporal_split,
)


def test_taiwan_compliance_guard() -> None:
    sample = "Operator contact: 0912-345-678, office 03-1234-5678, mail dev@example.org"
    detected, count, sanitized = TaiwanComplianceGuard.audit_and_redact(sample)
    assert detected is True
    assert count == 3
    assert "0912-345-678" not in sanitized
    assert "dev@example.org" not in sanitized


def test_block_temporal_split() -> None:
    assert verify_block_temporal_split([0.0, 1.0, 2.0], [10.0, 11.0], min_gap_s=5.0) is True
    assert verify_block_temporal_split([0.0, 1.0, 2.0], [2.2, 11.0], min_gap_s=5.0) is False


def test_preprocess_and_fusion() -> None:
    rgb = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    depth_mm = np.random.randint(500, 5000, size=(64, 64), dtype=np.uint16)
    lwir_raw = np.random.randint(10000, 15000, size=(64, 64), dtype=np.uint16)

    vis_t, aux_t = preprocess_rgbd_lwir_opencv(rgb, depth_mm, lwir_raw)
    assert vis_t.shape == (3, 64, 64)
    assert aux_t.shape == (2, 64, 64)

    model = CrossModalSE3Fusion(embed_dim=64, num_heads=4).eval()
    imu = torch.zeros((1, 6), dtype=torch.float32)
    with torch.inference_mode():
        out = model(vis_t.unsqueeze(0), aux_t.unsqueeze(0), imu)
    assert out.shape == (1, 16, 64)


def test_logit_adjusted_loss() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, -0.5]], dtype=torch.float32)
    targets = torch.tensor([2, 1], dtype=torch.long)
    priors = torch.tensor([0.90, 0.08, 0.02], dtype=torch.float32)
    loss = logit_adjusted_loss(logits, targets, priors, tau=1.0)
    assert torch.isfinite(loss).item() is True
    assert loss.item() > 0.0


if __name__ == "__main__":
    test_taiwan_compliance_guard()
    test_block_temporal_split()
    test_preprocess_and_fusion()
    test_logit_adjusted_loss()
    print("[PASS] All Python unit tests succeeded.")
