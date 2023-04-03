#include <cub/cub.cuh>   // or equivalently <cub/device/device_histogram.cuh>
#include<thrust/device_vector.h>
//#include<thrust/execution_policy.h>
#include<fstream>

void llenar_exponential_deviate(thrust::device_vector<float> &input)
{
	int N=input.size();
  	for(int i = 0; i < N; i++)
  	{
      		input[i] = -log(rand()/float(RAND_MAX));
		#ifdef DEBUG
      		std::cout << input[i] << " ";
		#endif
  	}
	#ifdef DEBUG
  	std::cout << std::endl;
	#endif
}


class histograma_cub
{
    private:
    // Declare, allocate, and initialize device-accessible pointers for input samples and
    // output histogram
    int      num_samples;    // e.g., 10
    float*   d_samples;      // e.g., [2.2, 6.0, 7.1, 2.9, 3.5, 0.3, 2.9, 2.0, 6.1, 999.5]
    int*     d_histogram;    // e.g., [ -, -, -, -, -, -, -, -]
    int      num_levels;     // e.g., 7       (seven level boundaries for six bins)
    float    lower_level;    // e.g., 0.0     (lower sample value boundary of lowest bin)
    float    upper_level;    // e.g., 12.0    (upper sample value boundary of upper bin)

    void*    d_temp_storage;
    size_t   temp_storage_bytes;

    int*     d_histogram_acum;    // e.g., [ -, -, -, -, -, -, -, -]
    int	     Nvectorsamples;
	
    public:

    void change_d_samples(float *d_samples_){d_samples=d_samples_;};

    histograma_cub(int _num_samples, int _num_levels, float _lower_level, float _upper_level, float *_d_samples):
    num_samples(_num_samples),num_levels(_num_levels),lower_level(_lower_level), upper_level(_upper_level),
    d_samples(_d_samples)	
    {
	    cudaMalloc(&d_histogram, sizeof(float)*num_levels);

	    // Determine temporary device storage requirements
	    d_temp_storage = NULL;
	    temp_storage_bytes = 0;
	    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
	        d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);
	    // Allocate temporary storage
	    cudaMalloc(&d_temp_storage, temp_storage_bytes);
	    cudaMalloc(&d_histogram_acum, sizeof(float)*num_levels);
	    reset();
    }

    ~histograma_cub(){
	cudaFree(d_temp_storage);
	cudaFree(d_histogram_acum);
	cudaFree(d_histogram);
    }

    void reset(){
	    thrust::device_ptr<int> d_histogram_acum_ptr(d_histogram_acum);
	    thrust::fill(d_histogram_acum_ptr,d_histogram_acum_ptr+num_levels,0.f);	
	    Nvectorsamples=0;	
    }

    histograma_cub(){
    	int _num_samples=100;
	int _num_levels=7;
    	float _lower_level=0.0;
    	float _upper_level=3.0;			
	histograma_cub(_num_samples, _num_levels,_lower_level,_upper_level,d_samples);
	Nvectorsamples++;
    }

    void compute_histogram()
    {
	// histograma instantaneo
	cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);

	// acumulacion
	using namespace thrust::placeholders;
	thrust::device_ptr<int> d_histogram_ptr(d_histogram);
	thrust::device_ptr<int> d_histogram_acum_ptr(d_histogram_acum);

  	thrust::transform(d_histogram_acum_ptr,d_histogram_acum_ptr+num_levels,
	d_histogram_ptr,d_histogram_acum_ptr,_1+_2);
	Nvectorsamples++;  		
    }	

    // imprime el bin ($1), el histograma acumulado ($2), y el ultimo histograma ($3)
    void print(std::ofstream &fout){

	thrust::device_ptr<int> d_his_ptr(d_histogram);
	thrust::device_ptr<int> d_his_acum_ptr(d_histogram_acum);

	for(int i=0;i<num_levels-1;i++){
		fout << lower_level + i*(upper_level-lower_level)/num_levels
		<< " " << d_his_acum_ptr[i]/Nvectorsamples 
		<< " " << d_his_ptr[i] 
		<< std::endl;
	}
    }

};

/*int main()
{
    // parametros para histograma	
    int num_samples=400;
    int num_levels=7;
    int lower_level=0.0;
    int upper_level=3.0;

    // aparentemente para inicializar el histograma hace falta tener ya el puntero a los datos...	
    thrust::device_vector<float> d_v_samples(num_samples,0);
    float *d_samples=thrust::raw_pointer_cast(d_v_samples.data());		

    // contruccion del histograma			
    histograma_cub H(num_samples,num_levels, lower_level, upper_level, d_samples);	

    // 100 vectors samples de num_samples cada uno
    for(int n=0;n<100;n++){
    	llenar_exponential_deviate(d_v_samples);

	// calcula y acumula histograma de d_samples
    	H.compute_histogram();
    }

    std::ofstream fout("out.dat");	
    H.print(fout);	

    // para graficarlo lueguito:
    system("gnuplot -p -e \"set tit ' histograma cub de -log(rand)';plot 'out.dat' u 1:2 w lp t 'acumulado', '' u 1:3 w lp t 'instantaneo';\" ");
}
*/

/*
int main1()
{
    // Declare, allocate, and initialize device-accessible pointers for input samples and
    // output histogram
    int      num_samples;    // e.g., 10
    float*   d_samples;      // e.g., [2.2, 6.0, 7.1, 2.9, 3.5, 0.3, 2.9, 2.0, 6.1, 999.5]
    int*     d_histogram;    // e.g., [ -, -, -, -, -, -, -, -]
    int      num_levels;     // e.g., 7       (seven level boundaries for six bins)
    float    lower_level;    // e.g., 0.0     (lower sample value boundary of lowest bin)
    float    upper_level;    // e.g., 12.0    (upper sample value boundary of upper bin)

    num_samples=100;
    thrust::device_vector<float> d_v_samples(num_samples);
    d_samples=thrust::raw_pointer_cast(d_v_samples.data());		
    llenar_exponential_deviate(d_v_samples);

    num_levels=7;
    lower_level=0.0;
    upper_level=3.0;			

    thrust::device_vector<int> d_v_histogram(num_levels);
    d_histogram=thrust::raw_pointer_cast(d_v_histogram.data());		


    // Determine temporary device storage requirements
    void*    d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Compute histograms
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);
    // d_histogram   <-- [1, 0, 5, 0, 3, 0, 0, 0];

    // el resultado en un file
    std::ofstream fout("out.dat");	
    for(int i=0;i<num_levels;i++){
	std::cout << d_v_histogram[i] << std::endl;
	fout << d_v_histogram[i] << std::endl;
    }	

    // para graficarlo lueguito:
    system("gnuplot -p -e \"set tit ' histograma de -log(rand)';plot 'out.dat' u 0:1 w lp t '';\" ");

    return 0;	
}*/
