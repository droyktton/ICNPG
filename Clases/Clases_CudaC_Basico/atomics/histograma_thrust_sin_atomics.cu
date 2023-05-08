#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <iostream>


//util para el profiling!
#include <nvToolsExt.h>

#define NUM_BINS 10
#define NUM_SAMPLES 1000000

int main()
{
  // Generate some random data
  const int N = NUM_SAMPLES;

  thrust::host_vector<float> h_data(N);
  thrust::generate(h_data.begin(), h_data.end(), rand);

  using namespace thrust::placeholders;
  thrust::transform(h_data.begin(), h_data.end(), h_data.begin(), _1*NUM_BINS/RAND_MAX);

  thrust::device_vector<int> data(h_data);

  nvtxRangePush("-----Histograma en Device-----");

  // Sort the data to group identical values together
  thrust::sort(data.begin(), data.end());

  // Count the number of occurrences of each value
  thrust::device_vector<int> counts(N);
  thrust::device_vector<int> keys(N);
  thrust::reduce_by_key(
                        data.begin(),data.end(),
                        thrust::make_constant_iterator(1),
                        keys.begin(),
                        counts.begin());
  nvtxRangePop();


  std::cout << std::endl;
  for(int n=0;n<NUM_BINS;n++){
    std::cout << keys[n]*1.0/NUM_BINS << " " << counts[n] << std::endl;
  }
}

