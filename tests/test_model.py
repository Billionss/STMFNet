import torch
import unittest
from layers.Temporal_Block import Attention_Block

class TestAttentionBlock(unittest.TestCase):
    def setUp(self):
        self.attention_block = Attention_Block(d_model=34, n_heads=8, dropout=0.1, activation="relu")

    def test_forward(self):
        x = torch.randn(64, 96, 34)
        output = self.attention_block(x)
        self.assertEqual(output.shape, (64, 96, 34))

if __name__ == '__main__':
    unittest.main()