import tempfile
import unittest
import subprocess
import sys
import os
from pathlib import Path
import torch
import main as lab


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        lab.block_size, lab.batch_size = 8, 2
        lab.n_embd, lab.n_head, lab.n_layer = 16, 2, 1
        lab.device, lab.max_iters, lab.eval_iters, lab.eval_interval = 'cpu', 1, 1, 1
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'corpus.txt'
        self.path.write_text('crop health and plant care.\n' * 50, encoding='utf-8')
        lab.prepare_dataset(self.path)

    def test_shifted_batch(self):
        x, y = lab.get_batch('train')
        self.assertEqual(tuple(x.shape), (2, 8))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_import_has_no_dataset_side_effect(self):
        env = dict(os.environ, PYTHONPATH=str(Path(lab.__file__).parent))
        result = subprocess.run([sys.executable, '-c', 'import main; assert main.train_data is None'], cwd=self.tmp.name, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [self.path])

    def test_small_dataset_has_clear_error(self):
        self.path.write_text('abc', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'splits'):
            lab.prepare_dataset(self.path)

    def test_unknown_prompt_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            lab.encode('☃')

    def test_generation_validates_options(self):
        model = lab.AdamLanguageModel(lab.vocab_size)
        for temperature in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                model.generate(torch.zeros((1, 1), dtype=torch.long), 1, temperature)

    def test_one_step_and_checkpoint(self):
        checkpoint = Path(self.tmp.name) / 'model.pt'
        model = lab.train(checkpoint)
        saved = torch.load(checkpoint, weights_only=True)
        self.assertEqual(saved['chars'], lab.chars)
        restored = lab.AdamLanguageModel(len(saved['chars']))
        restored.load_state_dict(saved['model'])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), saved['model'][key]))
        output = model.generate(torch.zeros((1, 1), dtype=torch.long), 2)
        self.assertEqual(tuple(output.shape), (1, 3))


if __name__ == '__main__':
    unittest.main()
