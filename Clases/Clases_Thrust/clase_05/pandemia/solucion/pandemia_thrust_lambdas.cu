/*
	Vamos a resolver el modelo SIR de epidemias
	
	dS/dt = -beta*S*I
	dI/dt = beta*S*I - gamma*I
	dR/dt = gamma*I

 	para N valores de beta
*/


// Poner los #include que hagan falta!
#include<thrust/device_vector.h>
#include<thrust/fill.h>
#include<thrust/sequence.h>

#define gamma	0.1  // tasa de recuperacion
#define Dt	0.1  // paso de tiempo


int main(void)
{
    int N=10;

    // Declarar y Alocar memoria para los arrays de device S, I, R y beta usando thrust 
    // ....
    thrust::device_vector<float> S(N);
    thrust::device_vector<float> I(N);
    thrust::device_vector<float> R(N);
    thrust::device_vector<float> beta(N);

    // Inicializar S[i]=0.999, I[i]=0.001, R[i]=0, y beta[i]=0.02+i*0.02 usando Thrust
    // ....
    thrust::fill(S.begin(),S.end(),0.999);
    thrust::fill(I.begin(),I.end(),0.001);
    thrust::fill(R.begin(),R.end(),0.0);

    thrust::sequence(beta.begin(),beta.end(),0.02,0.02);

    int ntot=5000;
	
    // loop de tiempo
    for(int n=0;n<ntot;n++){	

      // imprimir I[] en columnas
      for(int i=0;i<N;i++){
        std::cout << I[i] << "\t";
      }
      std::cout << "\n";

      thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(S.begin(), I.begin(), R.begin(),beta.begin())
        ),
        thrust::make_zip_iterator(
          thrust::make_tuple(S.end(), I.end(), R.end(),beta.end())
        ),
        [=] __device__ (auto tup)
        //[=] __device__ (thrust::tuple<float,float,float,float> tup)
        {
          float oldS=thrust::get<0>(tup);
          float oldI=thrust::get<1>(tup);
          float oldR=thrust::get<2>(tup);
          float b = thrust::get<3>(tup);

          thrust::get<0>(tup) = oldS - Dt*(b*oldS*oldI);
          thrust::get<1>(tup) = oldI + Dt*(b*oldS*oldI-gamma*oldI);
          thrust::get<2>(tup) = oldR - Dt*(gamma*oldI);
        }

      );
    }
}

/*
Para que ande bien esta version necesitamos cargar 

1) cuda/11.4.0   2) gcc/8.2.0

pero no esta soportado por gpushort
*/
