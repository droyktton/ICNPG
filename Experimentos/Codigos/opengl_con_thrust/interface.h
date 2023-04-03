#include <thrust/execution_policy.h>
#include <thrust/functional.h>

///////////////////////////////////////////////////////////
sistema *Sptr;
int paso;
float fieldmax, fieldmin;
int sqx, sqy;
float signo;
float hext;
//std::ifstream ftextin("tref_flip2.pbm.phi4");
std::ofstream outwall("wall.dat");

// rescalea la intensidad de la imagen segun la maxima amplitud del campo
struct absfloat: public thrust::unary_function<float,float>
{
  __host__ __device__
  float operator()(float x) { return (x>0)?(x):(-x); }
};
void rescale(){
  float *d = Sptr->dptr;
  //fieldmax = thrust::reduce(thrust::device, d, d+DIM*DIM, 0.f, thrust::maximum<float>());
  fieldmax = thrust::transform_reduce(thrust::device, d, d+DIM*DIM, absfloat(),0.f, thrust::maximum<float>());
}

// functor para mapear buffer de opengl al del sistema y viceversa (overloaded)
struct mapear_cuda_ogl_op
{
    float max, min;
    mapear_cuda_ogl_op(float _min, float _max):max(_max),min(_min){};

    __device__
    uchar4 operator()(float v)
    {
      uchar4 u4;
      u4.x = u4.y = u4.z = u4.w = 0;
      u4.y = (v>0)?(v*255/max):(0);
      u4.x = (v<0)?(-v*255/max):(0);
      //u4.y = int((v-min)*255/max);
      return u4;
    }
    __device__
    float operator()(uchar4 u)
    {

      return (u.y*max/255.f)-(u.x*max/255.f);
    }
};

void mapear_cuda_ogl(uchar4 *ptr,float *d){
    //float min=-3; float max=3;
    //max = thrust::reduce(thrust::device, d, d+DIM*DIM, 0.f, thrust::maximum<float>());
    thrust::transform(thrust::device, d, d+DIM*DIM, ptr, mapear_cuda_ogl_op(fieldmin,fieldmax));  
};
void mapear_ogl_cuda(float *d, uchar4 *ptr){
    float min=-3; float max=3;
    thrust::transform(thrust::device, ptr, ptr+DIM*DIM, d, mapear_cuda_ogl_op(min, max));  
};


// lo que pasa cada vez que opengl llama a draw
void change_pixels(){
  cudaGraphicsMapResources( 1, &resource, NULL ); 
  uchar4* devPtr; 
  size_t  size; 
  cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, resource); 

  Sptr->dynamics(paso);

  mapear_cuda_ogl(devPtr,Sptr->dptr);
  cudaGraphicsUnmapResources( 1, &resource, NULL ); 
};

void inicializar_variables_globales_sistema()
{
  paso=0;
  signo=1.0f;
  rescale();
}
////////////////////////////////////////////////

