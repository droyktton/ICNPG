#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iomanip>
#include <iterator>

#include <fstream>
#include <cstdlib>

//#define DEBUG

// This example illustrates several methods for computing a
// histogram [1] with Thrust.  We consider standard "dense"
// histograms, where some bins may have zero entries, as well
// as "sparse" histograms, where only the nonzero bins are
// stored.  For example, histograms for the data set
//    [2 1 0 0 2 2 1 1 1 1 4]
// which contains 2 zeros, 5 ones, and 3 twos and 1 four, is
//    [2 5 3 0 1]
// using the dense method and 
//    [(0,2), (1,5), (2,3), (4,1)]
// using the sparse method. Since there are no threes, the 
// sparse histogram representation does not contain a bin
// for that value.
//
// Note that we choose to store the sparse histogram in two
// separate arrays, one array of keys and one array of bin counts,
//    [0 1 2 4] - keys
//    [2 5 3 1] - bin counts
// This "structure of arrays" format is generally faster and
// more convenient to process than the alternative "array
// of structures" layout.
//
// The best histogramming methods depends on the application.
// If the number of bins is relatively small compared to the 
// input size, then the binary search-based dense histogram
// method is probably best.  If the number of bins is comparable
// to the input size, then the reduce_by_key-based sparse method 
// ought to be faster.  When in doubt, try both and see which
// is fastest.
//
// [1] http://en.wikipedia.org/wiki/Histogram


// simple routine to print contents of a vector
template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  #ifdef DEBUG
  typedef typename Vector::value_type T;
  std::cout << "  " << std::setw(20) << name << "  ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
  #endif	
}

// dense histogram using binary search
template <typename Vector1, 
          typename Vector2>
void dense_histogram(const Vector1& input,
                           Vector2& histogram)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);
    
  // print the initial data
  print_vector("initial data", data);
 
  // sort data to bring equal elements together
  thrust::sort(data.begin(), data.end());
  
  // print the sorted data
  print_vector("sorted data", data);

  // number of histogram bins is equal to the maximum value plus one
  IndexType num_bins = data.back() + 1;

  // resize histogram storage
  histogram.resize(num_bins);
  
  // find the end of each bin of values
  thrust::counting_iterator<IndexType> search_begin(0);
  thrust::upper_bound(data.begin(), data.end(),
                      search_begin, search_begin + num_bins,
                      histogram.begin());
  
  // print the cumulative histogram
  #ifdef DEBUG
  print_vector("cumulative histogram", histogram);
  #endif	

  // compute the histogram by taking differences of the cumulative histogram
  thrust::adjacent_difference(histogram.begin(), histogram.end(),
                              histogram.begin());

  // print the histogram
  print_vector("histogram", histogram);
}

// sparse histogram using reduce_by_key
template <typename Vector1,
          typename Vector2,
          typename Vector3>
void sparse_histogram(const Vector1& input,
                            Vector2& histogram_values,
                            Vector3& histogram_counts)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector3::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);
    
  // print the initial data
  print_vector("initial data", data);

  // sort data to bring equal elements together
  thrust::sort(data.begin(), data.end());
  
  // print the sorted data
  print_vector("sorted data", data);

  // number of histogram bins is equal to number of unique values (assumes data.size() > 0)
  IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                             data.begin() + 1,
                                             IndexType(1),
                                             thrust::plus<IndexType>(),
                                             thrust::not_equal_to<ValueType>());

  // resize histogram storage
  histogram_values.resize(num_bins);
  histogram_counts.resize(num_bins);
  
  // compact find the end of each bin of values
  thrust::reduce_by_key(data.begin(), data.end(),
                        thrust::constant_iterator<IndexType>(1),
                        histogram_values.begin(),
                        histogram_counts.begin());
  
  // print the sparse histogram
  print_vector("histogram values", histogram_values);
  print_vector("histogram counts", histogram_counts);
}


/* 
Una clase posible para histograma denso de floats
(muchas cuentas en un bin)
*/
class histograma_denso
{
	private:

	// numero de bins
	int nbins;

	// maximo y minimo de valores a histogramear
	float maximo, minimo;

	// histograma instantaneo
	thrust::device_vector<int> his;

	// histograma acumulado
	thrust::device_vector<int> his_acum;

	// numero de histogramas promediados
	int N_histo_samples;

	// quiza no se necesita, pero es comodo...
	thrust::device_vector<int> input;	

	public:
	// constructor:
	// INPUT: numero de bins, minimo y maximo valor, tamaño vector de input 
	histograma_denso(int _nbins, float _minimo, float _maximo, size_t Ninput):
	nbins(_nbins), minimo(_minimo), maximo(_maximo)
	{
		his_acum.resize(nbins);
		input.resize(Ninput);

		thrust::fill(his_acum.begin(),his_acum.end(),float(0.0));
		N_histo_samples=0;
	};

	histograma_denso()
	{
		maximo=1.0;
		minimo=0.0;
		his_acum.resize(100);		
	};

	// pone a cero el acumulador
	void reset(){
		thrust::fill(his_acum.begin(),his_acum.end(),float(0.0));
		N_histo_samples=0;
	}
	
	// imprime el bin ($1), el histograma acumulado ($2), y el ultimo histograma ($3)
	void print(std::ofstream &fout){
		for(int i=0;i<nbins;i++){
			fout << minimo + i*(maximo-minimo)/nbins 
			<< " " << his_acum[i]/N_histo_samples 
			<< " " << his[i] 
			<< std::endl;
		}
	}

	// supone input real en [0,1]
	void compute_dense_histogram(thrust::device_vector<float> &real_input)
	{

		/*using namespace thrust::placeholders;
		
		float fac=nbins/(maximo-minimo);

  		thrust::transform(real_input.begin(),real_input.end(),input.begin(),(_1-minimo)*fac);
  		dense_histogram(input, his);

  		thrust::transform(his_acum.begin(),his_acum.end(),his.begin(),his_acum.begin(),_1+_2);
		N_histo_samples++;
		*/
		compute_dense_histogram(thrust::raw_pointer_cast(real_input.data()), real_input.size());		
	}
	
	void compute_dense_histogram(float *real_input, size_t N)
	{
		using namespace thrust::placeholders;
		
		float fac=nbins/(maximo-minimo);

		thrust::device_ptr<float> real_input_ptr(real_input);

  		thrust::transform(real_input_ptr,real_input_ptr+N,input.begin(),(_1-minimo)*fac);
  		dense_histogram(input, his);

  		thrust::transform(his_acum.begin(),his_acum.end(),his.begin(),his_acum.begin(),_1+_2);
		N_histo_samples++;  		
	}

};

int main1(void)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 9);

  const int N = 40;
  const int S = 4;

  // generate random data on the host
  thrust::host_vector<int> input(N);
  for(int i = 0; i < N; i++)
  {
    int sum = 0;
    for (int j = 0; j < S; j++)
      sum += dist(rng);
    input[i] = sum / S;
  }

  // demonstrate dense histogram method
  {
    std::cout << "Dense Histogram" << std::endl;
    thrust::device_vector<int> histogram;
    dense_histogram(input, histogram);
  }
  

  // demonstrate sparse histogram method
  {
    std::cout << "Sparse Histogram" << std::endl;
    thrust::device_vector<int> histogram_values;
    thrust::device_vector<int> histogram_counts;

    sparse_histogram(input, histogram_values, histogram_counts);
  }
  
  // Note: 
  // A dense histogram can be converted to a sparse histogram
  // using stream compaction (i.e. thrust::copy_if).
  // A sparse histogram can be expanded into a dense histogram
  // by initializing the dense histogram to zero (with thrust::fill)
  // and then scattering the histogram counts (with thrust::scatter).

  return 0;
}


