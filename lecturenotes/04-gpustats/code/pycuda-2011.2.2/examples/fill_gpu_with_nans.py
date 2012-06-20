import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy

free_bytes, total_bytes = cuda.mem_get_info()
exp = 10
while True:
    fill_floats = free_bytes / 4 - (1<<exp)
    if fill_floats < 0:
        raise RuntimeError("couldn't find allocatable size")
    try:
        print "alloc", fill_floats
        ary = gpuarray.empty((fill_floats,), dtype=numpy.float32)
        break
    except:
        pass

    exp += 1

ary.fill(float("nan"))

print "filled %d out of %d bytes with NaNs" % (fill_floats*4, free_bytes)

