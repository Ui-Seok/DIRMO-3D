"""Batch alignment, left padding, coverage, and validation selection checks."""
import unittest
from types import SimpleNamespace

from dirmo.training.smdd_full import batch_losses, length_batches, selection_key
from dirmo.training.smdd_sft import completion_loss


class FullSmddTests(unittest.TestCase):
    def test_batched_loss_matches_individual_masked_losses_with_unequal_lengths(self):
        import torch

        class CausalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(19, 7)
                self.head = torch.nn.Linear(7, 19)

            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
                hidden = (self.embedding(input_ids) * attention_mask[..., None]).cumsum(dim=1)
                logits = self.head(hidden)
                return SimpleNamespace(logits=logits[:, logits_to_keep] if isinstance(logits_to_keep, torch.Tensor)
                                       else logits[:, -logits_to_keep:])

        torch.manual_seed(8)
        model = CausalModel()
        samples = [{"input_ids": [1, 2, 3, 4, 5], "prompt_length": 3, "target_ids": [4, 5]},
                   {"input_ids": [6, 7, 8, 9, 10, 11, 12, 13], "prompt_length": 5, "target_ids": [11, 12, 13]}]
        individual = torch.stack([completion_loss(model, s, "cpu") for s in samples])
        values, pooled = batch_losses(model, samples, "cpu", 0)
        self.assertTrue(torch.allclose(values, individual, atol=1e-6))
        expected = (individual[0] * 2 + individual[1] * 3) / 5
        self.assertTrue(torch.allclose(pooled, expected, atol=1e-6))
        pooled.backward()
        self.assertEqual(float(model.embedding.weight.grad[0].abs().sum()), 0)

    def test_length_buckets_cover_every_row_once(self):
        samples = [{"input_ids": list(range(20 + i)), "id": i} for i in range(53)]
        batches = length_batches(samples, 8, 42)
        self.assertEqual(sorted(s["id"] for b in batches for s in b), list(range(53)))
        self.assertTrue(all(1 <= len(b) <= 8 for b in batches))
        self.assertEqual(batches, length_batches(samples, 8, 42))

    def test_chemical_selection_does_not_choose_lower_loss_over_usable_edits(self):
        def report(passed, valid, loss):
            return {"fence_removed_diagnostic": {"counts": {"changed_and_local_passed": passed, "smiles_valid": valid}},
                    "completion_loss": loss}
        self.assertGreater(selection_key(report(3, 10, 0.9)), selection_key(report(0, 30, 0.1)))
        self.assertGreater(selection_key(report(3, 12, 0.9)), selection_key(report(3, 10, 0.1)))
        self.assertGreater(selection_key(report(3, 12, 0.3)), selection_key(report(3, 12, 0.5)))
        self.assertGreater(selection_key(report(3, 12, 0.3)), selection_key(report(3, 12, None)))


if __name__ == "__main__":
    unittest.main()
