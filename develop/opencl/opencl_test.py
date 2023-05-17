import numpy as np
import pyopencl as cl
from time import perf_counter

print(cl.get_platforms())

N = 50000000
a_np = np.random.rand(N).astype(np.float32)
b_np = np.random.rand(N).astype(np.float32)


t1 = perf_counter()
res_cpu = a_np + b_np
t1 = perf_counter() - t1

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prg = cl.Program(ctx, """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
knl = prg.sum  # Use this Kernel object for repeated calls
knl(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)

t0 = perf_counter()
cl.enqueue_copy(queue, res_np, res_g)
t0 = perf_counter() - t0



print('GPU:', t0*1e3)
print('CPU:', t1*1e3)

# Check on CPU with Numpy:
print(res_np - res_cpu)
##print(np.linalg.norm(res_np - (a_np + b_np)))
print(np.allclose(res_np, res_cpu))
