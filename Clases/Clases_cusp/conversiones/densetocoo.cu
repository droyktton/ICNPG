/*
Un ejemplo muy simple mostrando: 
-carga de una matriz en formato denso standard en el host
-impresion en pantalla de la matriz
-escritura en disco en formato matrix_market
-lectura de disco en formato matrix_market a formato coo en device
-impresion de la matriz de device en formato coo 
*/

#include <cusp/io/matrix_market.h>
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>

int main(void)
{
    // un ejemplo simple en formato 2d standard (denso) en host
    cusp::array2d<float, cusp::host_memory> A(3,4);
    A(0,0) = 10;  A(0,1) =  0;  A(0,2) = 20;  A(0,3) =  0;
    A(1,0) =  0;  A(1,1) = 30;  A(1,2) =  0;  A(1,3) = 40;
    A(2,0) = 50;  A(2,1) = 60;  A(2,2) = 70;  A(2,3) = 80;

    // la imprimo en pantalla para verificar...
    std::cout << "matriz A en formato denso" << std::endl;	
    cusp::print(A);

    // guardo en disco en MatrixMarket format
    std::cout << "chequear A.mtx en disco" << std::endl;	
    cusp::io::write_matrix_market_file(A, "A.mtx");

    // cargo A del disco en formato coo_matrix en device
    // implica copia host->device
    cusp::coo_matrix<int, float, cusp::device_memory> B;
    cusp::io::read_matrix_market_file(B, "A.mtx");

    // print B en formato coo, implica copia device->host
    std::cout << "matriz B en formato COO" << std::endl;	
    cusp::print(B);
    std::cout << "chequear B.mtx en disco" << std::endl;	
    cusp::io::write_matrix_market_file(B, "B.mtx");

    // conversion directa de A(densa) a C(coo_matrix) en device
    cusp::coo_matrix<int, float, cusp::device_memory> C(A);
    std::cout << "matriz C en formato COO (copia device-device)" << std::endl;	
    cusp::print(C);


    return 0;
}

