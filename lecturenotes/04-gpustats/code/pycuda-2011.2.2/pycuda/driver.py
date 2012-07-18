try:
    from pycuda._driver import *
except ImportError, e:
    if "_v2" in str(e):
        from warnings import warn
        warn("Failed to import the CUDA driver interface, with an error "
                "message indicating that the version of your CUDA header "
                "does not match the version of your CUDA driver.")
    raise

import numpy as np




CUDA_DEBUGGING = False




def set_debugging(flag=True):
    global CUDA_DEBUGGING
    CUDA_DEBUGGING = flag




class CompileError(Error):
    def __init__(self, msg, command_line, stdout=None, stderr=None):
        self.msg = msg
        self.command_line = command_line
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self):
        result = self.msg
        if self.command_line:
            try:
                result += "\n[command: %s]" % (" ".join(self.command_line))
            except Exception, e:
                print e
        if self.stdout:
            result += "\n[stdout:\n%s]" % self.stdout
        if self.stderr:
            result += "\n[stderr:\n%s]" % self.stderr

        return result




class ArgumentHandler(object):
    def __init__(self, ary):
        self.array = ary
        self.dev_alloc = None

    def get_device_alloc(self):
        if self.dev_alloc is None:
            self.dev_alloc = mem_alloc_like(self.array)
        return self.dev_alloc

    def pre_call(self, stream):
        pass

class In(ArgumentHandler):
    def pre_call(self, stream):
        if stream is not None:
            memcpy_htod(self.get_device_alloc(), self.array)
        else:
            memcpy_htod(self.get_device_alloc(), self.array)

class Out(ArgumentHandler):
    def post_call(self, stream):
        if stream is not None:
            memcpy_dtoh(self.array, self.get_device_alloc())
        else:
            memcpy_dtoh(self.array, self.get_device_alloc())

class InOut(In, Out):
    pass





def _add_functionality():

    def device_get_attributes(dev):
        result = {}

        for att_name in dir(device_attribute):
            if not att_name[0].isupper():
                continue

            att_id = getattr(device_attribute, att_name)

            try:
                att_value = dev.get_attribute(att_id)
            except LogicError, e:
                from warnings import warn
                warn("CUDA driver raised '%s' when querying '%s' on '%s'"
                        % (e, att_name, dev))
            else:
                result[att_id] = att_value

        return result

    def device___getattr__(dev, name):
        return dev.get_attribute(getattr(device_attribute, name.upper()))

    def function_param_set(func, *args):

        handlers = []

        arg_data = []
        format = ""
        for i, arg in enumerate(args):
            if isinstance(arg, np.number):
                arg_data.append(arg)
                format += arg.dtype.char
            elif isinstance(arg, (DeviceAllocation, PooledDeviceAllocation)):
                arg_data.append(int(arg))
                format += "P"
            elif isinstance(arg, ArgumentHandler):
                handlers.append(arg)
                arg_data.append(int(arg.get_device_alloc()))
                format += "P"
            elif isinstance(arg, np.ndarray):
                arg_data.append(arg)
                format += "%ds" % arg.nbytes
            else:
                try:
                    gpudata = np.intp(arg.gpudata)
                except AttributeError:
                    raise TypeError("invalid type on parameter #%d (0-based)" % i)
                else:
                    # for gpuarrays
                    arg_data.append(int(gpudata))
                    format += "P"

        from pycuda._pvt_struct import pack
        buf = pack(format, *arg_data)

        func._param_setv(0, buf)
        func._param_set_size(len(buf))

        return handlers

    def function_call(func, *args, **kwargs):
        grid = kwargs.pop("grid", (1,1))
        stream = kwargs.pop("stream", None)
        block = kwargs.pop("block", None)
        shared = kwargs.pop("shared", None)
        texrefs = kwargs.pop("texrefs", [])
        time_kernel = kwargs.pop("time_kernel", False)

        if kwargs:
            raise ValueError(
                    "extra keyword arguments: %s" 
                    % (",".join(kwargs.iterkeys())))

        if block is None:
            raise ValueError, "must specify block size"

        func._set_block_shape(*block)
        handlers = func._param_set(*args)
        if shared is not None:
            func.set_shared_size(shared)

        for handler in handlers:
            handler.pre_call(stream)

        for texref in texrefs:
            func.param_set_texref(texref)

        post_handlers = [handler
                for handler in handlers
                if hasattr(handler, "post_call")]

        if stream is None:
            if time_kernel:
                Context.synchronize()

                from time import time
                start_time = time()
            func._launch_grid(*grid)
            if post_handlers or time_kernel:
                Context.synchronize()

                if time_kernel:
                    run_time = time()-start_time

                for handler in post_handlers:
                    handler.post_call(stream)

                if time_kernel:
                    return run_time
        else:
            assert not time_kernel, "Can't time the kernel on an asynchronous invocation"
            func._launch_grid_async(grid[0], grid[1], stream)

            if post_handlers:
                for handler in post_handlers:
                    handler.post_call(stream)

    def function_prepare(func, arg_types, block=None, shared=None, texrefs=[]):
        from warnings import warn
        if block is not None:
            warn("setting the block size in Function.prepare is deprecated",
                    DeprecationWarning, stacklevel=2)
            func._set_block_shape(*block)

        if shared is not None:
            warn("setting the shared memory size in Function.prepare is deprecated",
                    DeprecationWarning, stacklevel=2)
            func._set_shared_size(shared)

        func.texrefs = texrefs

        func.arg_format = ""
        param_size = 0

        for i, arg_type in enumerate(arg_types):
            if isinstance(arg_type, type) and np is not None and np.number in arg_type.__mro__:
                func.arg_format += np.dtype(arg_type).char
            elif isinstance(arg_type, str):
                func.arg_format += arg_type
            else:
                func.arg_format += np.dtype(np.intp).char

        from pycuda._pvt_struct import calcsize
        func._param_set_size(calcsize(func.arg_format))

        return func

    def function_prepared_call(func, grid, block, *args, **kwargs):
        if isinstance(block, tuple):
            func._set_block_shape(*block)
        else:
            from warnings import warn
            warn("Not passing the block size to prepared_call is deprecated as of "
                    "version 2011.1.", DeprecationWarning, stacklevel=2)
            args = (block,) + args

        from pycuda._pvt_struct import pack
        func._param_setv(0, pack(func.arg_format, *args))

        for texref in func.texrefs:
            func.param_set_texref(texref)

        func._launch_grid(*grid)

    def function_prepared_timed_call(func, grid, block, *args, **kwargs):
        if isinstance(block, tuple):
            func._set_block_shape(*block)
        else:
            from warnings import warn
            warn("Not passing the block size to prepared_timed_call is deprecated as of "
                    "version 2011.1.", DeprecationWarning, stacklevel=2)
            args = (block,) + args

        from pycuda._pvt_struct import pack
        func._param_setv(0, pack(func.arg_format, *args))

        for texref in func.texrefs:
            func.param_set_texref(texref)

        start = Event()
        end = Event()

        start.record()
        func._launch_grid(*grid)
        end.record()

        def get_call_time():
            end.synchronize()
            return end.time_since(start)*1e-3

        return get_call_time

    def function_prepared_async_call(func, grid, block, stream, *args, **kwargs):
        if isinstance(block, tuple):
            func._set_block_shape(*block)
        else:
            from warnings import warn
            warn("Not passing the block size to prepared_async_call is deprecated as of "
                    "version 2011.1.", DeprecationWarning, stacklevel=2)
            args = (stream,) + args
            stream = block

        from pycuda._pvt_struct import pack
        func._param_setv(0, pack(func.arg_format, *args))

        for texref in func.texrefs:
            func.param_set_texref(texref)

        if stream is None:
            func._launch_grid(*grid)
        else:
            grid_x, grid_y = grid
            func._launch_grid_async(grid_x, grid_y, stream)

    def function___getattr__(self, name):
        if get_version() >= (2,2):
            return self.get_attribute(getattr(function_attribute, name.upper()))
        else:
            if name == "num_regs": return self._hacky_registers
            elif name == "shared_size_bytes": return self._hacky_smem
            elif name == "local_size_bytes": return self._hacky_lmem
            else:
                raise AttributeError("no attribute '%s' in Function" % name)

    def mark_func_method_deprecated(func):
        def new_func(*args, **kwargs):
            from warnings import warn
            warn("'%s' has been deprecated in version 2011.1. Please use "
                    "the stateless launch interface instead." % func.__name__[1:], 
                    DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        try:
            from functools import update_wrapper
        except ImportError:
            pass
        else:
            try:
                update_wrapper(new_func, func)
            except:
                # User won't see true signature. Oh well.
                pass

        return new_func

    Device.get_attributes = device_get_attributes
    Device.__getattr__ = device___getattr__
    Function._param_set = function_param_set
    Function.__call__ = function_call
    Function.prepare = function_prepare
    Function.prepared_call = function_prepared_call
    Function.prepared_timed_call = function_prepared_timed_call
    Function.prepared_async_call = function_prepared_async_call
    Function.__getattr__ = function___getattr__

    for meth_name in ["set_block_shape", "set_shared_size",
            "param_set_size", "param_set", "param_seti", "param_setf", "param_setv",
            "launch", "launch_grid", "launch_grid_async"]:
        setattr(Function, meth_name, mark_func_method_deprecated(
                getattr(Function, "_"+meth_name)))





_add_functionality()




# {{{ pagelocked numpy arrays

def pagelocked_zeros(shape, dtype, order="C", mem_flags=0):
    result = pagelocked_empty(shape, dtype, order, mem_flags)
    result.fill(0)
    return result




def pagelocked_empty_like(array, mem_flags=0):
    if array.flags.c_contiguous:
        order = "C"
    elif array.flags.f_contiguous:
        order = "F"
    else:
        raise ValueError("could not detect array order")

    return pagelocked_empty(array.shape, array.dtype, order, mem_flags)




def pagelocked_zeros_like(array, mem_flags=0):
    result = pagelocked_empty_like(array, mem_flags)
    result.fill(0)
    return result

# }}}

# {{{ aligned numpy arrays

def aligned_zeros(shape, dtype, order="C", alignment=4096):
    result = aligned_empty(shape, dtype, order, alignment)
    result.fill(0)
    return result




def aligned_empty_like(array, alignment=4096):
    if array.flags.c_contiguous:
        order = "C"
    elif array.flags.f_contiguous:
        order = "F"
    else:
        raise ValueError("could not detect array order")

    return aligned_empty(array.shape, array.dtype, order, alignment)




def aligned_zeros_like(array, alignment=4096):
    result = aligned_empty_like(array, alignment)
    result.fill(0)
    return result

# }}}




def mem_alloc_like(ary):
    return mem_alloc(ary.nbytes)




def to_device(bf_obj):
    bf = buffer(bf_obj)
    result = mem_alloc(len(bf))
    memcpy_htod(result, bf)
    return result




def dtype_to_array_format(dtype):
    if dtype == np.uint8:
        return array_format.UNSIGNED_INT8
    elif dtype == np.uint16:
        return array_format.UNSIGNED_INT16
    elif dtype == np.uint32:
        return array_format.UNSIGNED_INT32
    elif dtype == np.int8:
        return array_format.SIGNED_INT8
    elif dtype == np.int16:
        return array_format.SIGNED_INT16
    elif dtype == np.int32:
        return array_format.SIGNED_INT32
    elif dtype == np.float32:
        return array_format.FLOAT
    else:
        raise TypeError(
                "cannot convert dtype '%s' to array format" 
                % dtype)




def matrix_to_array(matrix, order, allow_double_hack=False):
    if order.upper() == "C":
        h, w = matrix.shape
        stride = 0
    elif order.upper() == "F":
        w, h = matrix.shape
        stride = -1
    else: 
        raise LogicError, "order must be either F or C"

    matrix = np.asarray(matrix, order=order)
    descr = ArrayDescriptor()

    descr.width = w
    descr.height = h

    if matrix.dtype == np.float64 and allow_double_hack:
        descr.format = array_format.SIGNED_INT32
        descr.num_channels = 2
    else:
        descr.format = dtype_to_array_format(matrix.dtype)
        descr.num_channels = 1

    ary = Array(descr)

    copy = Memcpy2D()
    copy.set_src_host(matrix)
    copy.set_dst_array(ary)
    copy.width_in_bytes = copy.src_pitch = copy.dst_pitch = \
            matrix.strides[stride]
    copy.height = h
    copy(aligned=True)

    return ary




def make_multichannel_2d_array(ndarray, order):
    """Channel count has to be the first dimension of the C{ndarray}."""

    descr = ArrayDescriptor()

    if order.upper() == "C":
        h, w, num_channels = ndarray.shape
        stride = 0
    elif order.upper() == "F":
        num_channels, w, h = ndarray.shape
        stride = 2
    else: 
        raise LogicError, "order must be either F or C"

    descr.width = w
    descr.height = h
    descr.format = dtype_to_array_format(ndarray.dtype)
    descr.num_channels = num_channels

    ary = Array(descr)

    copy = Memcpy2D()
    copy.set_src_host(ndarray)
    copy.set_dst_array(ary)
    copy.width_in_bytes = copy.src_pitch = copy.dst_pitch = \
            ndarray.strides[stride]
    copy.height = h
    copy(aligned=True)

    return ary




def bind_array_to_texref(ary, texref):
    texref.set_array(ary)
    texref.set_address_mode(0, address_mode.CLAMP)
    texref.set_address_mode(1, address_mode.CLAMP)
    texref.set_filter_mode(filter_mode.POINT)
    assert texref.get_flags() == 0




def matrix_to_texref(matrix, texref, order):
    bind_array_to_texref(matrix_to_array(matrix, order), texref)




def from_device(devptr, shape, dtype, order="C"):
    result = np.empty(shape, dtype, order)
    memcpy_dtoh(result, devptr)
    return result




def from_device_like(devptr, other_ary):
    result = np.empty_like(other_ary)
    memcpy_dtoh(result, devptr)
    return result
