import cupy as cp

dev=cp.cuda.runtime.getDevice()
props=cp.cuda.runtime.getDeviceProperties(dev)

print(f"Current device index: {dev}")
print(props['name'])


