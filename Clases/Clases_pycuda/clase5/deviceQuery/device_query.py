import pycuda.driver as cuda
import pycuda.autoinit

print("cantidad de devices =",cuda.Device.count())

(free,total) = cuda.mem_get_info()
print("Total Global memory: %.1f" % total)
print("Free Global memory:  %.1f" % free)


for devicenum in range(cuda.Device.count()):
    device=cuda.Device(devicenum)
    attrs=device.get_attributes()
    
    for key, value in attrs.items():
        print(key,"=", value)