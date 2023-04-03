// lee de pgm y lo guarda en un host_vector
void leepgm(std::ifstream &finp, thrust::host_vector<int> &hz)
{
	// avanzamos sobre el encabezado de pgm
	std::string nombre;
	std::string doscincocinco("255");
	do{
		std::getline(finp,nombre);
	}while(nombre.compare(doscincocinco)!=0);

	int value;
	for(int i=0;i<L*L;i++)
	{
		finp >> value;
		hz[i]=value;
	}
}

// lee un host vector y lo guarda en un pgm
void escribepgm(std::ofstream &fout, thrust::host_vector<int> &hz)
{
	fout << "P2\n" << L << " " << L << "\n255\n";
	for(int j=0;j<L*L;j++)
	{
		fout << hz[j] << std::endl;
	}
}


