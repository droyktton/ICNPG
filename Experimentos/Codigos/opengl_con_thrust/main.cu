#include<iostream>
#include<fstream>
#include "misistema.h"

std::ofstream outwall("wall.dat");
int main(){
	int L=512;
	sistema S(L);
	
	S.addcircle(L/2,L/2,L/5,1.0);

	std::ofstream fconf("configs.dat");
	//S.print_config(fconf);

	S.set_hext(0.012);

	for(int n=0;n<50;n++)
	{
		S.print_config(fconf);
		S.dynamics(10000);
		S.detect_wall(outwall);

		std::cout << n << std::endl;
	}
		
	return 0;
}
