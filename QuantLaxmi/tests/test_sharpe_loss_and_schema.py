"""Tests for Bug #20 (Sharpe loss zero-std fix) and Bug #25 (schema versioning).

Bug #20: sharpe_loss must return a differentiable penalty (not a detached zero)
when strategy returns have zero std.

Bug #25: EventEnvelope must carry schema_version through serialization.
"""

from __future__ import annotations

import json
import math
import unittest


class TestSharpeLossZeroStd(unittest.TestCase):
    """Bug #20: Sharpe loss with zero-std returns must produce useful gradients."""

    def setUp(self):
        try:
            import torch
            self.torch = torch
            self.has_torch = True
        except ImportError:
            self.has_torch = False

    def test_zero_std_produces_nonzero_gradient_momentum_tfm(self):
        """momentum_tfm sharpe_loss with zero-std returns gives non-zero grad."""
        if not self.has_torch:
            self.skipTest("PyTorch not available")

        from quantlaxmi.models.ml.tft.momentum_tfm import sharpe_loss

        torch = self.torch
        # Positions that require grad
        positions = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
        # All-zero returns → zero std
        returns = torch.tensor([0.0, 0.0, 0.0, 0.0])

        loss = sharpe_loss(positions, returns)

        # Loss should be non-zero (penalty for not taking varied positions)
        self.assertNotEqual(loss.item(), 0.0, "Loss should not be zero for zero-std returns")

        # Must be able to backprop
        loss.backward()

        # Gradient must exist and be non-zero
        self.assertIsNotNone(positions.grad)
        self.assertFalse(
            torch.all(positions.grad == 0).item(),
            "Gradient should be non-zero to provide useful gradient flow",
        )

    def test_zero_std_produces_nonzero_gradient_x_trend(self):
        """x_trend sharpe_loss with zero-std returns gives non-zero grad."""
        if not self.has_torch:
            self.skipTest("PyTorch not available")

        from quantlaxmi.models.ml.tft.x_trend import sharpe_loss

        torch = self.torch
        positions = torch.tensor([0.3, 0.3, 0.3, 0.3], requires_grad=True)
        returns = torch.tensor([0.0, 0.0, 0.0, 0.0])

        loss = sharpe_loss(positions, returns)

        self.assertNotEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(positions.grad)
        self.assertFalse(torch.all(positions.grad == 0).item())

    def test_normal_returns_sharpe_momentum_tfm(self):
        """momentum_tfm sharpe_loss with normal returns computes correct Sharpe."""
        if not self.has_torch:
            self.skipTest("PyTorch not available")

        from quantlaxmi.models.ml.tft.momentum_tfm import sharpe_loss

        torch = self.torch
        positions = torch.tensor([1.0, 1.0, -1.0, 1.0], requires_grad=True)
        returns = torch.tensor([0.01, -0.005, -0.02, 0.015])

        loss = sharpe_loss(positions, returns)

        # Strategy returns: [0.01, -0.005, 0.02, 0.015]
        strat_ret = positions.detach() * returns
        expected_mean = strat_ret.mean().item()
        expected_std = strat_ret.std(correction=1).item()
        expected_sharpe = (expected_mean / expected_std) * math.sqrt(252)

        self.assertAlmostEqual(loss.item(), -expected_sharpe, places=4)

        # Must produce gradients
        loss.backward()
        self.assertIsNotNone(positions.grad)

    def test_normal_returns_sharpe_x_trend(self):
        """x_trend sharpe_loss with normal returns computes correct Sharpe."""
        if not self.has_torch:
            self.skipTest("PyTorch not available")

        from quantlaxmi.models.ml.tft.x_trend import sharpe_loss

        torch = self.torch
        positions = torch.tensor([1.0, -1.0, 0.5, -0.5], requires_grad=True)
        returns = torch.tensor([0.02, 0.01, -0.01, 0.005])

        loss = sharpe_loss(positions, returns)

        strat_ret = positions.detach() * returns
        expected_mean = strat_ret.mean().item()
        expected_std = strat_ret.std(correction=1).item()
        expected_sharpe = (expected_mean / expected_std) * math.sqrt(252)

        self.assertAlmostEqual(loss.item(), -expected_sharpe, places=4)

        loss.backward()
        self.assertIsNotNone(positions.grad)

    def test_constant_returns_zero_std(self):
        """Constant non-zero returns with same positions → zero std → penalty path."""
        if not self.has_torch:
            self.skipTest("PyTorch not available")

        from quantlaxmi.models.ml.tft.momentum_tfm import sharpe_loss

        torch = self.torch
        positions = torch.tensor([0.8, 0.8, 0.8, 0.8], requires_grad=True)
        returns = torch.tensor([0.01, 0.01, 0.01, 0.01])

        loss = sharpe_loss(positions, returns)

        # std of strategy_returns = 0 → penalty path
        # penalty = -abs(positions).mean() * 0.01 = -0.8 * 0.01 = -0.008
        self.assertAlmostEqual(loss.item(), -0.008, places=5)

        loss.backward()
        self.assertIsNotNone(positions.grad)
        self.assertFalse(torch.all(positions.grad == 0).item())


class TestEventEnvelopeSchemaVersion(unittest.TestCase):
    """Bug #25: EventEnvelope must have schema_version in serialized output."""

    def test_envelope_has_schema_version(self):
        """EventEnvelope includes schema_version field."""
        from quantlaxmi.core.events.envelope import EventEnvelope

        env = EventEnvelope(
            ts="2025-01-01T00:00:00.000000Z",
            seq=1,
            run_id="test-run",
            event_type="signal",
            source="test",
        )
        self.assertEqual(env.schema_version, "1.0")

    def test_schema_version_in_serialized_output(self):
        """schema_version appears in serialized JSON."""
        from quantlaxmi.core.events.envelope import EventEnvelope
        from quantlaxmi.core.events.serde import serialize_envelope

        env = EventEnvelope(
            ts="2025-01-01T00:00:00.000000Z",
            seq=1,
            run_id="test-run",
            event_type="signal",
            source="test",
        )
        serialized = serialize_envelope(env)
        d = json.loads(serialized)

        self.assertIn("schema_version", d)
        self.assertEqual(d["schema_version"], "1.0")

    def test_deserialization_with_schema_version(self):
        """Deserialization reads schema_version correctly."""
        from quantlaxmi.core.events.serde import (
            serialize_envelope,
            deserialize_envelope,
        )
        from quantlaxmi.core.events.envelope import EventEnvelope

        env = EventEnvelope(
            ts="2025-01-01T00:00:00.000000Z",
            seq=1,
            run_id="test-run",
            event_type="signal",
            source="test",
        )
        serialized = serialize_envelope(env)
        restored = deserialize_envelope(serialized)

        self.assertEqual(restored.schema_version, "1.0")

    def test_deserialization_missing_schema_version_backward_compat(self):
        """Old JSON without schema_version defaults to '1.0'."""
        from quantlaxmi.core.events.serde import deserialize_envelope

        # Simulate old-format JSON without schema_version
        old_json = json.dumps({
            "ts": "2025-01-01T00:00:00.000000Z",
            "seq": 1,
            "run_id": "test-run",
            "event_type": "signal",
            "source": "test",
            "strategy_id": "",
            "symbol": "",
            "payload": {},
        }, sort_keys=True, separators=(",", ":"))

        env = deserialize_envelope(old_json)
        self.assertEqual(env.schema_version, "1.0")

    def test_create_factory_includes_schema_version(self):
        """EventEnvelope.create() includes schema_version."""
        from quantlaxmi.core.events.envelope import EventEnvelope

        env = EventEnvelope.create(
            seq=1,
            run_id="test-run",
            event_type="signal",
            source="test",
            payload={"foo": "bar"},
        )
        self.assertEqual(env.schema_version, "1.0")

    def test_roundtrip_with_schema_version(self):
        """Serialize → deserialize → serialize is bit-identical with schema_version."""
        from quantlaxmi.core.events.envelope import EventEnvelope
        from quantlaxmi.core.events.serde import (
            serialize_envelope,
            deserialize_envelope,
            roundtrip_stable,
        )

        env = EventEnvelope(
            ts="2025-01-01T00:00:00.000000Z",
            seq=42,
            run_id="run-abc",
            event_type="order",
            source="engine",
            payload={"price": 100.5},
        )
        self.assertTrue(roundtrip_stable(env))


if __name__ == "__main__":
    unittest.main()
