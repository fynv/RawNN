import os
import numpy as np

from cffi import FFI
ffi  = FFI()

ffi.cdef("""
    void* facedetector_create();
    void facedetector_destroy(void* ptr);
    void facedetector_run(void* ptr, const unsigned char* rgb128);
    int facedetector_num_faces(void* ptr);
    void facedetector_get_results(void* ptr, float* result);
""")

if os.name == 'nt':
    Native = ffi.dlopen('RawNN.dll')
elif os.name == "posix":
    Native = ffi.dlopen('libRawNN.so')


class FaceDetector:
    def __init__(self):
        self.cptr = Native.facedetector_create()

    def __del__(self):
        Native.facedetector_destroy(self.cptr)     
        
    def Run(self, arr):
        Native.facedetector_run(self.cptr, ffi.cast("unsigned char*", arr.__array_interface__['data'][0]))

    def GetResults(self):
        num_faces = Native.facedetector_num_faces(self.cptr)
        res = np.empty((num_faces, 17), dtype=np.float32)
        Native.facedetector_get_results(self.cptr, ffi.cast("float*", res.__array_interface__['data'][0]))
        return res


