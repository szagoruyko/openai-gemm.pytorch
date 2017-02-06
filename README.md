openai-gemm.pytorch
========

PyTorch bindings for openai-gemm.

<https://github.com/openai/openai-gemm>


## Installation

Clone original openai-gemm and add it to PYTHONPATH,
install pycuda:

```
pip install pycuda
```

and follow instructions to install PyTorch on <http://pytorch.org>

No `neon` installation needed.

## Usage

The library defines `matmul` function similar to the one that
works with neon: <https://github.com/openai/openai-gemm/blob/master/openai_gemm.py#L14>,
which instead of neon matrices takes `torch.cuda.FloatTensor` or `torch.cuda.HalfTensor`
as A, B and C.
