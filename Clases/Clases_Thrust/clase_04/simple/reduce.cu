#include<thrust/reduce.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<cstdio>

int main()
{
	thrust::host_vector<float> hx( 1000000 );

	thrust::generate(hx.begin(),hx.end(),rand);

	thrust::device_vector<float> dx=hx;

	float sum=thrust::reduce(dx.begin(),dx.end());

	printf("suma=%f\n",sum);

	return 0;
}
