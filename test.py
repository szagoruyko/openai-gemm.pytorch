import unittest
import torch
from openai_gemm_pytorch import matmul


class TestMatMul(unittest.TestCase):
    def testNN(self):
        a = torch.randn(5,4).cuda()
        b = torch.randn(4,7).cuda()

        c = torch.Tensor(5,7).cuda()
        matmul(a, b, c)

        self.assertLess((c - a.mm(b)).abs().max(), 1e-6)

    def testNT(self):
        a = torch.randn(5,4).cuda()
        b = torch.randn(7,4).cuda().t()

        c = torch.Tensor(5,7).cuda()
        matmul(a, b, c)

        self.assertLess((c - a.mm(b)).abs().max(), 1e-6)

if __name__ == '__main__':
    unittest.main()
