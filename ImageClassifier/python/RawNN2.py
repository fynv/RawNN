import os
import numpy as np

from cffi import FFI
ffi  = FFI()

ffi.cdef("""
    void* classifier_create();
    void classifier_destroy(void* ptr);
    float classifier_run(void* ptr, const unsigned char* rgb128); 
""")

if os.name == 'nt':
    Native = ffi.dlopen('RawNN2.dll')
elif os.name == "posix":
    Native = ffi.dlopen('libRawNN2.so')


class Classifier:
    def __init__(self):
        self.cptr = Native.classifier_create()

    def __del__(self):
        Native.classifier_destroy(self.cptr)     
        
    def Run(self, arr):
        return Native.classifier_run(self.cptr, ffi.cast("unsigned char*", arr.__array_interface__['data'][0]))
