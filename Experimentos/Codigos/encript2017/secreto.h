// TODO: escriba su/sus functors aqui...
struct mifunctor{
	int *offsets;
	int *senial;
	mifunctor(int * _off, int * _sen):offsets(_off),senial(_sen){};
	__device__
	int operator()(int tid)
	{
		int x=tid%L;
		int y=int(tid/L);
		int xx = (x+offsets[y])%L;		
		int index = xx + L*y;
		return senial[index];
	}
};

// TODO: escriba su kernel/kernels aqui...
__global__ 
void mikernel(int *offsets, int *senialenc, int *senialdes)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int x=tid%L;
	int y=int(tid/L);
	int xx = (x+offsets[y])%L;		
	int index = xx + L*y;
	senialdes[tid]=senialenc[index];
}

void encriptar(char *name)
{
	std::ifstream finpgm(name);

	// avanzamos sobre el encabezado de pgm
	std::string nombre;
	std::string doscincocinco("255");
	do{
		std::getline(finpgm,nombre);
	}while(nombre.compare(doscincocinco)!=0);

	// cargamos el resto de la imagen
	int N=L*L;
	thrust::host_vector<int> h_inp(N);
	for(int n=0;n<L*L;n++)
	{
		int value;
		finpgm >> value;
		h_inp[n]=value;
	}


	// una opcion es elegir offsets random
	//std::srand ( unsigned ( std::time(0) ) );
	thrust::host_vector<int> hoffsets(L);
	//for(int j=0;j<L;j++) hoffsets[j]=int(L*(std::rand()*1.0/RAND_MAX))%L;
	// otra es esta
	//for(int j=0;j<L;j++) hoffsets[j]=j*12345;

	for(int j=0;j<L;j++) hoffsets[j]=thrust::reduce(h_inp.begin()+j*L,h_inp.begin()+(j+1)*L);

	// vector encriptado
	thrust::host_vector<int> hz(N);
	for(int j=0;j<L;j++)
	{
		for(int i=0;i<L;i++)
		{
			int value = h_inp[i+j*L];
			int n=(i+hoffsets[j])%L+j*L;
			hz[n]=value;
		}		
	}


	// guardamos en un file el vector encriptado
	char nom[50];
	sprintf(nom,"%s_encripted.pgm",name);
	std::ofstream fout(nom);
	fout << "P2\n" << L << " " << L << "\n255\n";
	for(int j=0;j<N;j++)
	{
		fout << hz[j] << "\n";
	}
	fout.close();
}


