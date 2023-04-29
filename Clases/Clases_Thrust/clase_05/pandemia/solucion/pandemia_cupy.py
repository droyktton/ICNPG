import cupy as cp
import numpy as np
import cupyx

# define the kernel as a string
SIRkernel = cp.RawKernel(r'''
  extern "C" __global__ 
  void modeloSIR(float *S, float *I, float *R, float *beta, float gamma, float dt, int N)
  {

    int i = threadIdx.x + blockIdx.x*blockDim.x;
    //printf("i=%d, gamma=%f N=%d",i, gamma, N);

    float newS, newI, newR;
    float oldS, oldI, oldR;

    oldS=S[i];
    oldI=I[i];
    oldR=R[i];

    dt=0.1; gamma=0.1; 
    float b=beta[i];

    //b=0.1;

    if(i<N){

      newS = oldS - dt * b * oldS * oldI;

      newI = oldI + dt * (b*oldS*oldI - gamma*oldI);

      newR = oldR + dt * (gamma*oldI);

      S[i]=newS;
      I[i]=newI;
      R[i]=newR;
    }

}''', 'modeloSIR')


N = 10

gamma = 0.1  # tasa de recuperacion
dt = 0.1  # paso de tiempo

# Declarar y Alocar memoria para los arrays de device S, I, R y beta usando CuPy
# ....
S = cp.zeros(N, dtype=cp.float32)
I = cp.zeros(N, dtype=cp.float32)
R = cp.zeros(N, dtype=cp.float32)
beta = cp.zeros(N, dtype=cp.float32)

# Inicializar S[i]=0.999, I[i]=0.001, R[i]=0, y beta[i]=0.02+i*0.02 usando CuPy
# ....
S.fill(0.999)
I.fill(0.001)
R.fill(0.0)
beta.fill(0.1)

print("S=",S, len(beta))
print("I=",I, len(beta))
print("R=",R, len(beta))
print("beta=",beta, len(beta))

ntot = 5000

f = open("data.csv", "w")
    
h_I = np.empty(N, dtype=np.float32)
#h_I=I.get()
#np.savetxt(f, h_I.reshape(1, -1), delimiter='\t', fmt='%f')

# loop de tiempo
for p in range(ntot):
  # imprimir I[] en columnas
    # ...
    h_I=I.get()
    #print(h_I)

    #print(h_I)
    np.savetxt(f, h_I.reshape(1, -1), delimiter='\t', fmt='%f')

    block_size = 32
    grid_size = (N + block_size - 1)//block_size

    #print(block_size,grid_size)
    #print(gamma, Dt, N)

    # Llamar al kernel de actualizacion de S[],I[],R[]
    SIRkernel((grid_size,), (block_size,), (S, I, R, beta, gamma, dt, N))

    #print(t)
    #cp.cuda.Device().synchronize()
    
