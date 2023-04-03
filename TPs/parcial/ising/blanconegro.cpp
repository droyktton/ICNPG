#include<iostream>
#include<cstdlib>

bool blanconegro(int n, int L, int shift)
{
	return ((n+shift)%2==(n/L)%2);
}
bool blanconegro2(int n, int L, int shift)
{
	int nx=n%L;
	int ny=int(n/L);
	return ((nx+ny)%2==0);
}

int main(int argc, char **argv)
{
	int shift = (argc>1)?(atoi(argv[1])%2):(0);
	int L=(argc>2)?(atoi(argv[2])):(4);
	int N=L*L;

	for(int n=0;n<N;n++)
	{
		{
			int nx=n%L;
			int ny=int(n/L);
			std::cout << n << ", fila=" << ny << " izq=";
			std::cout << (nx-1+L)%L  + ny*L << " der=";
			std::cout << (nx+1+L)%L + ny*L << " arr=";
			std::cout << nx + ((ny+1)%L)*L << " aba=";
			std::cout << nx + ((ny-1+L)%L)*L << " color=";
			std::cout << blanconegro(n,L,shift) << " vs " ;
			std::cout << blanconegro2(n,L,shift) << std::endl;
		}

	};
}
