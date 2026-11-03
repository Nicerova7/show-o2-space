"""CPU checks for direction estimation and Show-o2 intervention hooks."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import torch

from experiments.custom_utils import (
    add_direction, build_steering_vector, capture_pair, count_denoising_evaluations,
    edit_component, estimate_direction, prepare_pair, repe_direction, steer_denoising,
)


class Block(torch.nn.Module):
    def forward(self, hidden):
        return (hidden,)


class Decoder(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.embedding = torch.nn.Embedding(64, 3)

    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids)
        for block in self.layers:
            hidden = block(hidden)[0]
        return hidden


class Denoiser(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.showo = SimpleNamespace(model=decoder)
        self.layers = decoder.layers
        self.dtype = torch.float32

    def forward(self, text_tokens, *, modality_positions):
        hidden = torch.ones(2, 12, 3, dtype=self.dtype)
        for block in self.layers:
            hidden = block(hidden)[0]
        return hidden


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) % 63 + 1 for char in text]}


class SteeringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        layers = torch.nn.ModuleList([Block(), Block()])
        decoder = Decoder(layers).eval().requires_grad_(False)
        self.adapter = SimpleNamespace(
            layers=layers, model=Denoiser(decoder), device=torch.device("cpu"),
            dtype=torch.float32, autocast=nullcontext, tokenizer=Tokenizer(),
            token_ids={"bos_id": 0},
        )
        self.vectors = {0: torch.tensor([2., 0., 0.]), 1: torch.tensor([0., 3., 0.])}
        self.positions = torch.tensor([[[5, 4]], [[2, 4]]])
        self.tokens = torch.zeros(2, 12, dtype=torch.long)

    def run_forward(self):
        return self.adapter.model(self.tokens, modality_positions=self.positions)

    def test_last_token_build_and_group_means(self):
        positives, negatives, differences = [], [], []
        for pos, neg in [("two", "one"), ("two poems", "one poem")]:
            captured = capture_pair(self.adapter, prepare_pair(self.adapter, pos, neg))
            result = build_steering_vector(captured)
            positives.append(result["positive"][0][0])
            negatives.append(result["negative"][0][0])
            differences.append(result["vectors"][0])
        positive, negative = torch.stack(positives), torch.stack(negatives)
        torch.testing.assert_close(torch.stack(differences).mean(0), estimate_direction(positive, negative))
        self.assertFalse(self.adapter.layers[0]._forward_hooks)

    def test_add_component_and_callable_math(self):
        hidden = torch.tensor([[3., 4., 5.]])
        vector = torch.tensor([2., 0., 0.])
        torch.testing.assert_close(add_direction(hidden, vector, alpha=.5), torch.tensor([[4., 4., 5.]]))
        torch.testing.assert_close(edit_component(hidden, vector, alpha=-1), torch.tensor([[0., 4., 5.]]))
        torch.testing.assert_close(edit_component(hidden, vector, alpha=1), torch.tensor([[6., 4., 5.]]))

    def test_repe_uses_pair_variation_and_unit_norm(self):
        differences = torch.tensor([[8., 1., 0.], [10., 1., 0.], [14., 1., 0.]])
        torch.testing.assert_close(repe_direction(differences), torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(repe_direction(-differences), torch.tensor([-1., 0., 0.]))
        self.assertEqual(repe_direction(differences).norm().item(), 1.)

    def test_modalities_cfg_and_special_positions(self):
        with steer_denoising(
            self.adapter, vectors=self.vectors, total_evaluations=1, layers=0,
            text=dict(tokens="last", operation="add", alpha=.5),
            visual=dict(tokens="all", operation="component", alpha=-1),
        ) as trace:
            result = self.run_forward()
        expected = torch.ones(2, 12, 3)
        expected[0, 3, 0] = 2
        expected[0, 6:9, 0] = 0
        torch.testing.assert_close(result, expected)
        self.assertEqual(trace["evaluations"], 1)
        self.assertEqual(len(trace["injections"]), 2)
        torch.testing.assert_close(trace["preview"]["0:text"]["before"], torch.ones(3))
        torch.testing.assert_close(trace["preview"]["0:text"]["after"], torch.tensor([2., 1., 1.]))

    def test_component_preserves_dtype_under_autocast(self):
        hidden = torch.tensor([[3., 4., 5.]], dtype=torch.bfloat16)
        vector = torch.tensor([2., 1., 3.], dtype=torch.bfloat16)
        unit = torch.nn.functional.normalize(vector.float(), dim=-1)
        expected = hidden.float() - (hidden.float() @ unit).unsqueeze(-1) * unit
        with torch.autocast("cpu", dtype=torch.bfloat16):
            edited = edit_component(hidden, vector, alpha=-1)
        self.assertEqual(edited.dtype, hidden.dtype)
        torch.testing.assert_close(edited, expected.to(hidden), rtol=0, atol=0)

    def test_bfloat16_hook_assignment(self):
        self.adapter.model.dtype = torch.bfloat16
        with torch.autocast("cpu", dtype=torch.bfloat16), steer_denoising(
            self.adapter, vectors=self.vectors, total_evaluations=1, layers=0,
            text=dict(tokens="all", operation="component", alpha=-1),
        ):
            result = self.run_forward()
        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(result[0, 1:4, 0], torch.zeros(3, dtype=torch.bfloat16))
        torch.testing.assert_close(result[1], torch.ones(12, 3, dtype=torch.bfloat16))

        def float_output(hidden, vector, *, alpha):
            return hidden.float() + alpha * vector.float()

        with steer_denoising(
            self.adapter, vectors=self.vectors, total_evaluations=1, layers=0,
            text=dict(tokens="all", operation=float_output, alpha=1),
        ):
            result = self.run_forward()
        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(result[0, 1:4, 0], torch.full((3,), 3., dtype=torch.bfloat16))

    def test_each_layer_uses_its_own_vector(self):
        with steer_denoising(
            self.adapter, vectors=self.vectors, total_evaluations=1,
            visual=dict(tokens="first", operation="add", alpha=1),
        ):
            result = self.run_forward()
        torch.testing.assert_close(result[0, 6], torch.tensor([3., 4., 1.]))
        torch.testing.assert_close(result[0, 7], torch.ones(3))

    def test_iteration_schedules(self):
        for selection, expected in [("first", [0]), ("last", [3]), (slice(0, 2), [0, 1]), ("all", [0, 1, 2, 3])]:
            with self.subTest(selection=selection), steer_denoising(
                self.adapter, vectors=self.vectors, total_evaluations=4,
                layers="last", iterations=selection,
                visual=dict(tokens=[0, 2], operation="add", alpha=1),
            ) as trace:
                for _ in range(4):
                    self.run_forward()
            self.assertEqual([row["iteration"] for row in trace["injections"]], expected)
            self.assertEqual(trace["evaluations"], 4)

    def test_zero_control_and_custom_operation(self):
        baseline = self.run_forward()
        with steer_denoising(
            self.adapter, vectors=self.vectors, total_evaluations=1,
            visual=dict(tokens="all", operation="add", alpha=0),
        ):
            torch.testing.assert_close(self.run_forward(), baseline, rtol=0, atol=0)

        def multiply(hidden, vector, *, alpha):
            return hidden * alpha

        with steer_denoising(
            self.adapter, vectors=self.vectors, total_evaluations=1, layers=0,
            visual=dict(tokens="last", operation=multiply, alpha=2),
        ):
            result = self.run_forward()
        torch.testing.assert_close(result[0, 8], torch.full((3,), 2.))
        torch.testing.assert_close(result[1], torch.ones(12, 3))

    def test_counter_and_cleanup(self):
        with count_denoising_evaluations(self.adapter) as trace:
            self.run_forward()
            self.run_forward()
        self.assertEqual(trace["evaluations"], 2)
        self.assertFalse(self.adapter.model._forward_pre_hooks)
        self.assertFalse(self.adapter.layers[0]._forward_hooks)

        class InterruptedRun(Exception):
            pass

        with self.assertRaises(InterruptedRun):
            with steer_denoising(self.adapter, vectors=self.vectors, total_evaluations=1):
                raise InterruptedRun
        self.assertFalse(self.adapter.model._forward_pre_hooks)
        self.assertFalse(self.adapter.layers[0]._forward_hooks)
