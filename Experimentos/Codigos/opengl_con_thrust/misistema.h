#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <cassert>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#define Dt 0.01


//#ifdef DISORDER
typedef float REAL;
#define AMPDIS	0.25
#define SEED 	12345678 // global seed RNG (quenched noise)
/* counter-based random numbers */
// http://www.thesalmons.org/john/random123/releases/1.06/docs/
#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG
typedef r123::Philox4x32 RNG4; // particular counter-based RNG


// para generar números aleatorios gausianos a partir de dos uniformes
// http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
__device__
REAL box_muller(RNG::ctr_type r_philox)
{
	// transforma el philox number a dos uniformes en (0,1]
 	REAL u1 = u01_open_closed_32_53(r_philox[0]);
  	REAL u2 = u01_open_closed_32_53(r_philox[1]);

  	REAL r = sqrt( -2.0*log(u1) );
  	REAL theta = 2.0*M_PI*u2;
	return r*sin(theta);    			
}
//#endif

// force functor
struct forceop
{
	int l;
	float * d;
	float * dis;
	float hext;
	forceop(float *_d, int _l, float _hext, float *_dis):d(_d),l(_l),hext(_hext),dis(_dis)
	{};

    __device__ 
    float operator()(int i)
    {
    	int x=i%l;
    	int y=int(i/l);

    	// periodic boundaries
    	int yp1 = (y+1)%l;
    	int ym1 = (y-1+l)%l;
    	int xp1 = (x+1)%l;
    	int xm1 = (x-1+l)%l;

    	int arriba= x+l*ym1;
    	int abajo= x+l*yp1;
    	int derecha= xp1 + l*y;
    	int izquierda=xm1 + l*y;
    	int centro = x+l*y;

    	float d_arriba = d[arriba];
    	float d_abajo = d[abajo];
    	float d_derecha = d[derecha];
    	float d_izquierda = d[izquierda];
	float d_centro = d[centro];

    	float laplacian = d_arriba + d_abajo + d_derecha + d_izquierda - 4.0*d_centro;
	float phi4force = d_centro - d_centro*d_centro*d_centro;// V = d_centro^4/4 - d_centro^2/2

	float RBdisorder=1.0;

	#ifdef DISORDER
	// random number generator
	RNG rng;       
	// keys and counters 
    	RNG::ctr_type c={{}};
    	RNG::key_type k={{}};
	RNG::ctr_type r;
	c[1]=uint32_t(SEED);
	c[0]=uint32_t(i);
	k[0]=i; 
	r = rng(c, k);
	#ifdef HALFDISORDER
	if(y>l*0.5) RBdisorder=1.0+AMPDIS*box_muller(r);
	#else
	RBdisorder=1.0+AMPDIS*box_muller(r);
	#endif
	#endif // DISORDER

	#ifdef SMOOTHDISORDER
	RBdisorder=1.0+dis[centro];	
	#endif

	#ifdef HUEVOS
	RBdisorder = 1.0 + 0.5*cos(2.*M_PI*x*10./l)*cos(2.*M_PI*y*10./l);
	#endif	

	// segmento que ancla, la derivada de phi se anula 
	#ifdef CULITOEFFECT
	if(x==int(l*0.5) && y>l*0.45 && y<l*0.65) {
		laplacian= d_arriba + d_abajo - 2.0*d_centro;
		//RBdisorder=0.1;
	}
	#endif		

	return laplacian + RBdisorder*phi4force + hext;
    }
};


// solo para testear la correctitud de los indices...
void test(int i, int l){

    	int x=i%l;
    	int y=int(i/l);

    	int arriba=    x + l*((y-1+l)%l);
    	int abajo=     x + l*((y+1)%l);
    	int derecha=   (x+1)%l + l*y;
    	int izquierda= (x-1+l)%l + l*y;
    	int centro =   x+l*y;

    	std::cout << x << " " << y << std::endl;
		std::cout << "\t\t" << arriba << "\t\t" << std::endl;	
		std::cout << izquierda << "\t\t" << centro << "\t\t" << derecha << "\t\t" << std::endl;	
		std::cout << "\t\t" << abajo << "\t\t" << std::endl;	
}

struct initial_condition
{
	int l;
	float *h;
	initial_condition(float *_h, int _l):h(_h),l(_l){};

    __device__
    float operator()(int i)
    {
    	int x=i%l;
    	int y=int(i/l);
    	
    	return  (sinf(x*4.0*2.0*M_PI/l)*sinf(y*4.0*2.0*M_PI/l))+
				(sinf(x*7.0*2.0*M_PI/l)*sinf(y*7.0*2.0*M_PI/l))+
				(sinf(x*14.0*2.0*M_PI/l)*cosf(y*14.0*2.0*M_PI/l));				
    }
};

// para generar números aleatorios gausianos a partir de dos uniformes
// http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
/*__device__
REAL box_muller(RNG::ctr_type r_philox)
{
	// transforma el philox number a dos uniformes en (0,1]
 	REAL u1 = u01_open_closed_32_53(r_philox[0]);
  	REAL u2 = u01_open_closed_32_53(r_philox[1]);

  	REAL r = sqrt( -2.0*log(u1) );
  	REAL theta = 2.0*M_PI*u2;
	return r*sin(theta);    			
}
*/

// returns the contribution of mode q==(kx,ky) at point i==(x,y)
// for(i) V[i]==V(x,y)=reduce(qmin, qmax, random_modes_fill(ptr,L,i)); 
/*struct random_modes_fill
{
	int l;
	float *h;
	random_modes_fill(float *_h, int _l):h(_h),l(_l){};

    __device__
    float operator()(int i)
    {
    	int x=i%l;
    	int y=int(i/l);

    	RNG4 rng;       
	// keys and counters 
    	RNG4::ctr_type c={{}};
    	RNG4::key_type k={{}};
		RNG4::ctr_type r;

		k[0]=uint32_t(SEED);; 

    	float sum=0.0;
    	int qmin=1;
    	int qmax=64;

    	for(int qx=qmin; qx< qmax; qx++){
    	for(int qy=qmin; qy< qmax; qy++){

			c[0]=uint32_t(qx);
			c[1]=uint32_t(qy);

			r = rng(c, k);

			// box muller
			float u1 = u01_open_closed_32_53(r[0]);
  			float u2 = u01_open_closed_32_53(r[1]);
			float amplitude=sqrtf( -2.0*logf(u1) )*sinf(2.0*M_PI*u2); // == gaussian RN   
			//amplitude=1;

			float phase=u01_open_closed_32_53(r[2])*2.0*M_PI;	

			//float angle=u01_open_closed_32_53(r[3])*2.0*M_PI;
			//float qx=2*M_PI*q*cosf(angle)/l;
			//float qy=2*M_PI*q*sinf(angle)/l;

			sum+= amplitude*cosf(phase + qx*x + qy*y );	
    	}}

    	return sum/((qmax-qmin)*(qmax-qmin));	 	
    }
};
*/

struct is_wall
{
    int l;
    float *h;
    is_wall(float *_h, int _l):h(_h),l(_l){};

    __device__
    bool operator()(int i)
    {
    	int x=i%l;
    	int y=int(i/l);

    	int arriba=    x + l*((y-1+l)%l);
    	int abajo=     x + l*((y+1)%l);
    	int derecha=   (x+1)%l + l*y;
    	int izquierda= (x-1+l)%l + l*y;
    	int centro =   x+l*y;
	
	if(h[arriba]*h[centro]<0 || h[abajo]*h[centro]<0 ||  h[derecha]*h[centro]<0 ||  h[izquierda]*h[centro]<0)	
	//if(h[arriba]*h[centro]<0 || h[derecha]*h[centro]<0)	
	return 1;
	else return 0;
    }
};

#include "randomsurface.h"
class sistema{
	private:
		thrust::device_ptr<float> d_color; //tambien podria ser un float * a device...
		thrust::device_ptr<float> d_force; //tambien podria ser un float * a device...
		thrust::device_ptr<float> d_disorder; //tambien podria ser un float * a device...

		thrust::device_ptr<int> indices_wall; // guarda la posicion de la pared

		int Lat; // el espacio es un cuadrado [0,Lat][0,Lat]
		int Lat2;

		float hext;

	public:
		float *dptr;
		float *dis_ptr;

		sistema(int l){
			Lat=l;
			Lat2=Lat*Lat;

			d_color = thrust::device_malloc<float>(Lat2);
			dptr = thrust::raw_pointer_cast(&d_color[0]);

			d_force = thrust::device_malloc<float>(Lat2);
			d_disorder = thrust::device_malloc<float>(Lat2);

			indices_wall = thrust::device_malloc<int>(Lat*10);

			hext=0.0;
			
			dis_ptr = thrust::raw_pointer_cast(&d_disorder[0]);

			randomsurface(1, Lat/10, d_disorder, Lat, Lat);

			//thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(Lat2),d_disorder,random_modes_fill(dis_ptr,Lat));
			
			reinicializar();
		}

		~sistema(){
			thrust::device_free(d_color);
			thrust::device_free(d_force);
		}

		void detect_wall(std::ofstream &fout){
			//dptr = thrust::raw_pointer_cast(&d_color[0]);
			int nwall = int
			(
				thrust::copy_if
				(
					thrust::make_counting_iterator(0),
					thrust::make_counting_iterator(Lat2),
					indices_wall,is_wall(dptr,Lat)
				) 
				-indices_wall
			);
			std::cout << "wall points=" << nwall << std::endl;

			using namespace thrust::placeholders;
			int area1 = thrust::count_if(thrust::device, dptr, dptr+Lat2, _1<0);
			int area2 = Lat2 - area1;

			fout << "# " << nwall << " " << area1 << " " << area2 << std::endl;

			thrust::host_vector<int> wallpoints(nwall);
			thrust::copy(indices_wall,indices_wall+nwall,wallpoints.begin());
			for(int i=0;i<nwall;i++){
				    	int x=wallpoints[i]%Lat;
				    	int y=int(wallpoints[i]/Lat);
					fout << x-Lat*0.5 << " " << y-Lat*0.5 << " " << atan2(y-Lat*0.5, x-Lat*0.5) << std::endl;
			}
			fout << std::endl << std::endl;
		}

		void reinicializar()
		{
			// combinacion de modos de fouriers 	
			/*thrust::transform
			(
				thrust::make_counting_iterator(0),
				thrust::make_counting_iterator(Lat2),
				d_color,
				initial_condition(dptr,Lat)
			);*/

			// random
			/*using namespace thrust::placeholders; 	
			thrust::host_vector<float> h_color(Lat2);
			thrust::generate(h_color.begin(),h_color.end(),rand);
			thrust::transform(h_color.begin(),h_color.end(), h_color.begin(),_1/RAND_MAX-0.5);
			thrust::copy(h_color.begin(), h_color.end(), d_color);
			*/

			// uniforme
			thrust::fill(d_color, d_color+Lat2, -1.0);
		};

		void addsquare(int x0, int y0, int L, float signo)
		{
			for(int i=x0;i<x0+L;i++)
			{
				for(int j=y0;j<y0+L;j++){
					int x=i%Lat;
					int y=j%Lat;
					//d_color[(x+Lat*y)]= (x==x0 || x==x0+L-1 || y==y0 || y==y0+L-1 )?(0):(signo);
					d_color[(x+Lat*y)]= (signo);
				}
			}

		};

		void addcircle(int x0, int y0, int L, float signo)
		{
			for(int i=x0-L;i<x0+L;i++)
			{
				for(int j=y0-L;j<y0+L;j++){
					int x=i%Lat;
					int y=j%Lat;
					if( (i-x0)*(i-x0)+(j-y0)*(j-y0) < L*L )
					d_color[(x+Lat*y)]= (signo);
				}
			}

		};

		void addtext(int x0, int y0,float signo, int Npix)
		{
			std::ifstream ftextin("tref_flip.pbm.phi4");
			int n0=x0+y0*Lat;
			float z;
			for(int i=0;i<Npix;i++)
			{
				ftextin >> z;
				//d_color[n0+i]= (z==0)?(-1):(1);
				if(i>512*512/2) d_color[n0+i]= (z==0)?(-1):(1);
				else d_color[n0+i]= (z==0)?(1):(-1);				
			}	
		};

		void set_hext(float _hext){
			hext=_hext;		
		}

		void dynamics(int trun)
		{
			using namespace thrust::placeholders;

			// hace trun steps de la dinamica
			for(int j=0;j<trun;j++)
			{
				// calcula la fuerza
				thrust::transform(
					thrust::make_counting_iterator(0),
					thrust::make_counting_iterator(Lat2),
					d_force,
					forceop(dptr,Lat,hext,dis_ptr)
				);

				// paso de euler
				thrust::transform(d_color,d_color+Lat2,d_force,d_color,_1+Dt*_2);
			}
		}
	
		void print_config(std::ofstream &fout){
			for(int i=0;i<Lat;i++){
				for(int j=0;j<Lat;j++){
					fout << d_color[j+i*Lat] << " ";
				}
				fout << "\n";
			}
			fout << "\n" << std::endl;
		}
};
