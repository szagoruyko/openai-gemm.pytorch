import pycuda.autoinit
import torch
from openai_gemm import _get_gemm_kernel, _get_bench_data

def is_transposed(t):
    return not (t.stride(1) == 1 and t.stride(0) != 0)

def matmul(A, B, C, alpha=1.0, beta=0.0, stream=None, bench=False):
    """
        C = alpha * A   . B   + beta * C
        C = alpha * A.T . B   + beta * C
        C = alpha * A   . B.T + beta * C
        C = alpha * A.T . B.T + beta * C

        bench: return benchmark data for all available tiles + cublas
    """

    if isinstance(C, torch.cuda.FloatTensor):
        prefix = "s"
    elif isinstance(C, torch.cuda.HalfTensor):
        prefix = "h"
    else:
        raise TypeError("Only floating point dot currently supported.")

    # (m,n) = (m,k) . (k,n)
    m = A.size(0)
    n = B.size(1)
    k = A.size(1)
    assert m == C.size(0)
    assert n == C.size(1)
    assert k == B.size(0)

    # Extract the operations and contiguous dimension sizes (cda, cdb, cdc).
    # Note that these can be the same as from the shape unless the non-contiguous dimension is sliced.
    # One dimension must be contiguous (DRAM efficiency demands this).
    # Note that the strides here do not include the datatype size as they would in numpy.
    # A transpose op (.T) on a GPUTensor reverses the shape and strides then flags the tensor as transposed (is_trans=True) -
    #    The underlying data is unchanged.
    if is_transposed(A):
         opA  = 'T'
         cda  = A.stride(1)
         assert A.stride(0) == 1
    else:
         opA  = 'N'
         cda  = A.stride(0)
         assert A.stride(1) == 1

    if is_transposed(B):
         opB  = 'T'
         cdb  = B.stride(1)
         assert B.stride(0) == 1
    else:
         opB  = 'N'
         cdb  = B.stride(0)
         assert B.stride(1) == 1

    cdc  = C.stride(0)
    assert C.stride(1) == 1

    op = opA + opB

    # get and autotune the kernel selection
    kernel, params, dynamic_shared = _get_gemm_kernel(prefix, op, cda, cdb, cdc, m, n, k)

    # bind dynamic params
    params[2:8] = (stream, C.data_ptr(), A.data_ptr(), B.data_ptr(), alpha, beta)

    # call the kernel
    kernel.prepared_async_call(*params, shared_size=dynamic_shared)

    # unbind dynamic params
    params[2:8] = (None,) * 6

    # return benchmark data if requested
    if bench:
        return _get_bench_data()[(prefix, op, cda, cdb, cdc, m, n, k)]

    return C
