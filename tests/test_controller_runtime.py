"""Tests for resolving response/action runtime settings from controller bundles."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ncpu.self_optimizing.controller_bundle import (
    ControllerBundle,
    ControllerComponentConfig,
    save_controller_bundle,
)
from ncpu.self_optimizing.controller_runtime import (
    load_bundle_latent_action_policy,
    load_bundle_latent_halt_policy,
    load_bundle_latent_memory_updater,
    resolve_controller_runtime,
)


class TestControllerRuntime(unittest.TestCase):
    def test_resolve_runtime_uses_bundle_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(
                        provider="hf_local",
                        model="response-adapter",
                        temperature=0.05,
                        base_url="http://response.example",
                        request_timeout=33.0,
                        max_tokens=256,
                        provider_kwargs={"fast_weights_rank": 8},
                    ),
                    action=ControllerComponentConfig(
                        provider="hf_local",
                        model="action-adapter",
                        temperature=0.01,
                        base_url="http://action.example",
                        request_timeout=9.0,
                        max_tokens=32,
                        provider_kwargs={"foo": "bar"},
                    ),
                ),
                bundle_path,
            )

            runtime = resolve_controller_runtime(
                controller_bundle_path=str(bundle_path),
                provider_name=None,
                model=None,
                temperature=None,
                base_url=None,
                request_timeout=None,
                default_provider="local",
                default_temperature=0.0,
                default_base_url="http://localhost:11434",
                default_request_timeout=120.0,
            )

        self.assertEqual(runtime.provider_name, "hf_local")
        self.assertEqual(runtime.model, "response-adapter")
        self.assertEqual(runtime.temperature, 0.05)
        self.assertEqual(runtime.base_url, "http://response.example")
        self.assertEqual(runtime.request_timeout, 33.0)
        self.assertEqual(runtime.max_tokens, 256)
        self.assertEqual(runtime.provider_kwargs, {"fast_weights_rank": 8})
        self.assertEqual(runtime.action_provider_name, "hf_local")
        self.assertEqual(runtime.action_model, "action-adapter")
        self.assertEqual(runtime.action_temperature, 0.01)
        self.assertEqual(runtime.action_base_url, "http://action.example")
        self.assertEqual(runtime.action_request_timeout, 9.0)
        self.assertEqual(runtime.action_max_tokens, 32)
        self.assertEqual(runtime.action_provider_kwargs, {"foo": "bar"})
        self.assertIsNone(runtime.latent_action_head_path)
        self.assertEqual(runtime.latent_action_head_config, {})
        self.assertIsNone(runtime.latent_memory_head_path)
        self.assertEqual(runtime.latent_memory_head_config, {})
        self.assertIsNone(runtime.latent_halt_head_path)
        self.assertEqual(runtime.latent_halt_head_config, {})

    def test_resolve_runtime_allows_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(provider="hf_local", model="response-adapter"),
                    action=ControllerComponentConfig(provider="hf_local", model="action-adapter"),
                ),
                bundle_path,
            )

            runtime = resolve_controller_runtime(
                controller_bundle_path=str(bundle_path),
                provider_name="local",
                model="override-model",
                temperature=0.3,
                base_url="http://override.example",
                request_timeout=77.0,
                action_provider_name="openai",
                action_model="override-action",
                action_temperature=0.4,
                action_base_url="http://action-override.example",
                action_request_timeout=55.0,
                max_tokens=512,
                provider_kwargs={"alpha": 1},
                action_max_tokens=99,
                action_provider_kwargs={"beta": 2},
                default_provider="local",
                default_temperature=0.0,
                default_base_url="http://localhost:11434",
                default_request_timeout=120.0,
            )

        self.assertEqual(runtime.provider_name, "local")
        self.assertEqual(runtime.model, "override-model")
        self.assertEqual(runtime.temperature, 0.3)
        self.assertEqual(runtime.base_url, "http://override.example")
        self.assertEqual(runtime.request_timeout, 77.0)
        self.assertEqual(runtime.max_tokens, 512)
        self.assertEqual(runtime.provider_kwargs, {"alpha": 1})
        self.assertEqual(runtime.action_provider_name, "openai")
        self.assertEqual(runtime.action_model, "override-action")
        self.assertEqual(runtime.action_temperature, 0.4)
        self.assertEqual(runtime.action_base_url, "http://action-override.example")
        self.assertEqual(runtime.action_request_timeout, 55.0)
        self.assertEqual(runtime.action_max_tokens, 99)
        self.assertEqual(runtime.action_provider_kwargs, {"beta": 2})

    def test_resolve_runtime_includes_relative_latent_action_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            head_path = Path(tmpdir) / "latent_action_head.pt"
            head_path.write_bytes(b"stub")
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(provider="hf_local", model="response-adapter"),
                    latent_action_head_path="latent_action_head.pt",
                    latent_action_head_config={"hidden_dim": 32, "dropout": 0.1},
                ),
                bundle_path,
            )

            runtime = resolve_controller_runtime(
                controller_bundle_path=str(bundle_path),
                provider_name=None,
                model=None,
                temperature=None,
                base_url=None,
                default_provider="local",
                default_temperature=0.0,
                default_base_url="http://localhost:11434",
            )

        self.assertEqual(runtime.latent_action_head_path, str(head_path.resolve()))
        self.assertEqual(runtime.latent_action_head_config["hidden_dim"], 32)

    def test_resolve_runtime_includes_relative_latent_halt_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            head_path = Path(tmpdir) / "latent_halt_head.pt"
            head_path.write_bytes(b"stub")
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(provider="hf_local", model="response-adapter"),
                    latent_halt_head_path="latent_halt_head.pt",
                    latent_halt_head_config={"hidden_dim": 24, "dropout": 0.2},
                ),
                bundle_path,
            )

            runtime = resolve_controller_runtime(
                controller_bundle_path=str(bundle_path),
                provider_name=None,
                model=None,
                temperature=None,
                base_url=None,
                default_provider="local",
                default_temperature=0.0,
                default_base_url="http://localhost:11434",
            )

        self.assertEqual(runtime.latent_halt_head_path, str(head_path.resolve()))
        self.assertEqual(runtime.latent_halt_head_config["hidden_dim"], 24)

    def test_resolve_runtime_includes_relative_latent_memory_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            head_path = Path(tmpdir) / "latent_memory_head.pt"
            head_path.write_bytes(b"stub")
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(provider="hf_local", model="response-adapter"),
                    latent_memory_head_path="latent_memory_head.pt",
                    latent_memory_head_config={"hidden_dim": 40, "output_dim": 16, "dropout": 0.05},
                ),
                bundle_path,
            )

            runtime = resolve_controller_runtime(
                controller_bundle_path=str(bundle_path),
                provider_name=None,
                model=None,
                temperature=None,
                base_url=None,
                default_provider="local",
                default_temperature=0.0,
                default_base_url="http://localhost:11434",
            )

        self.assertEqual(runtime.latent_memory_head_path, str(head_path.resolve()))
        self.assertEqual(runtime.latent_memory_head_config["hidden_dim"], 40)

    def test_load_bundle_latent_action_policy_uses_bundle_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            head_path = Path(tmpdir) / "latent_action_head.pt"
            head_path.write_bytes(b"stub")
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(provider="hf_local", model="response-adapter"),
                    latent_action_head_path="latent_action_head.pt",
                    latent_action_head_config={"hidden_dim": 32, "dropout": 0.1},
                ),
                bundle_path,
            )

            with mock.patch("ncpu.self_optimizing.latent_action_policy.load_latent_action_head", return_value=mock.Mock(config=mock.Mock())) as loader:
                policy = load_bundle_latent_action_policy(
                    controller_bundle_path=str(bundle_path),
                    device="cpu",
                )

        self.assertIsNotNone(policy)
        loader.assert_called_once()

    def test_load_bundle_latent_halt_policy_uses_bundle_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            head_path = Path(tmpdir) / "latent_halt_head.pt"
            head_path.write_bytes(b"stub")
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(provider="hf_local", model="response-adapter"),
                    latent_halt_head_path="latent_halt_head.pt",
                    latent_halt_head_config={"hidden_dim": 24, "dropout": 0.2},
                ),
                bundle_path,
            )

            with mock.patch("ncpu.self_optimizing.latent_halt_policy.load_latent_halt_head", return_value=mock.Mock(config=mock.Mock())) as loader:
                policy = load_bundle_latent_halt_policy(
                    controller_bundle_path=str(bundle_path),
                    device="cpu",
                )

        self.assertIsNotNone(policy)
        loader.assert_called_once()

    def test_load_bundle_latent_memory_updater_uses_bundle_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "controller_bundle.json"
            head_path = Path(tmpdir) / "latent_memory_head.pt"
            head_path.write_bytes(b"stub")
            save_controller_bundle(
                ControllerBundle(
                    name="demo",
                    response=ControllerComponentConfig(provider="hf_local", model="response-adapter"),
                    latent_memory_head_path="latent_memory_head.pt",
                    latent_memory_head_config={"hidden_dim": 40, "output_dim": 16, "dropout": 0.05},
                ),
                bundle_path,
            )

            with mock.patch(
                "ncpu.self_optimizing.latent_memory_head.load_latent_memory_head",
                return_value=mock.Mock(config=mock.Mock()),
            ) as loader:
                updater = load_bundle_latent_memory_updater(
                    controller_bundle_path=str(bundle_path),
                    device="cpu",
                )

        self.assertIsNotNone(updater)
        loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
