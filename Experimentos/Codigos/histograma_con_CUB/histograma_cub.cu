#include "histograma_cub.h"


struct mi_sistema{
    	thrust::device_vector<float> d_v_samples;
	float *d_samples;
	int num_samples; // longitud del vector 
	histograma_cub H;

	mi_sistema(int num_samples_=4000, int num_levels_=21, int lower_level_=0.0, int upper_level_=3.0):
	H(num_samples_,num_levels_, lower_level_, upper_level_, NULL)
	{
	    d_v_samples.resize(num_samples_);
	    d_samples=thrust::raw_pointer_cast(d_v_samples.data());
	    H.change_d_samples(d_samples);
	    
	};	

	void rellenar(){
		llenar_exponential_deviate(d_v_samples);
	}
	
	void histogramear(){
		H.compute_histogram();
	};
	
	void printhistograma(std::ofstream &fout){
		H.print(fout);
	}	
};




int main()
{
    // parametros para histograma	
    int num_samples=4000; // longitud del vector 
    int num_levels=21;    // numero de bins+1
    int lower_level=0.0; // minimo a histogramear
    int upper_level=3.0; // maximo a histogramear

    mi_sistema T(num_samples, num_levels, lower_level, upper_level);

    for(int n=0;n<100;n++){
    	T.rellenar();
    	T.histogramear(); 
    }
    
    // imprime el instantaneo y acumulado en el file "out.dat"
    std::ofstream fout("out.dat");	
    T.printhistograma(fout);	

    // para graficarlo lueguito:
    system("gnuplot -p -e \"set tit ' histograma cub de -log(rand)';plot 'out.dat' u 1:2 w lp t 'acumulado', '' u 1:3 w lp t 'instantaneo';\" ");

    return 0;    	
}



int main0()
{
    // parametros para histograma	
    int num_samples=4000; // longitud del vector 
    int num_levels=21;    // numero de bins+1
    int lower_level=0.0; // minimo a histogramear
    int upper_level=3.0; // maximo a histogramear

    // aparentemente para inicializar el histograma hace falta tener ya el puntero a los datos...	
    thrust::device_vector<float> d_v_samples(num_samples);
    float *d_samples=thrust::raw_pointer_cast(d_v_samples.data());		

    // contruccion del histograma (toma puntero a device de los datos una vez nomas)			
    histograma_cub H(num_samples,num_levels, lower_level, upper_level, d_samples);	

    // 100 vectors samples de num_samples cada uno
    for(int n=0;n<100;n++){
    	llenar_exponential_deviate(d_v_samples);

	// calcula y acumula histograma de d_samples
        // (el puntero a device de los datos modificados ta lo tiene la clase)
    	H.compute_histogram();
    }
  
    // imprime el instantaneo y acumulado en el file "out.dat"
    std::ofstream fout("out.dat");	
    H.print(fout);	

    // para graficarlo lueguito:
    system("gnuplot -p -e \"set tit ' histograma cub de -log(rand)';plot 'out.dat' u 1:2 w lp t 'acumulado', '' u 1:3 w lp t 'instantaneo';\" ");

    return 0;    	
}


