// include the csr_matrix header file
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>
#include <cusp/print.h>
#include <cusp/transpose.h>
#include <cusp/elementwise.h>
#include <cusp/graph/connected_components.h>
#include <cusp/graph/vertex_coloring.h>
#include <cusp/multiply.h>

template<typename Array2d>
void make_adjacency_matrix(Array2d &A)
{
  // initialize matrix entries on host
  // conectados con el 1
  A(1-1,1-1)=1;
  A(1-1,2-1)=1;
  A(1-1,5-1)=1;

  // conectados con el 2
  A(2-1,3-1)=1;
  A(2-1,5-1)=1;

  // conectados con el 3
  A(3-1,4-1)=1;

  // conectados con el 4
  A(4-1,5-1)=1;
  A(4-1,6-1)=1;

  // conectados con el 7
  A(7-1,8-1)=1;
  A(7-1,8-1)=1;

  // compute the transpose
  cusp::array2d<int, cusp::host_memory> At;
  cusp::transpose(A, At);
  //cusp::print(At);
  cusp::add(At,A,A);
  A(1-1,1-1)=1;
}

template<typename MemorySpace, typename MatrixType>
void CC(const MatrixType& G)
{
    typedef typename MatrixType::index_type IndexType;
    typedef cusp::csr_matrix<IndexType,IndexType,MemorySpace> GraphType;

    GraphType G_cc(G);
    cusp::array1d<IndexType,MemorySpace> components(G.num_rows);
    
    //timer t;
    size_t num_components = cusp::graph::connected_components(G_cc, components);
    //std::cout << "CC time : " << t.milliseconds_elapsed() << " (ms)." << std::endl;
    std::cout << "numero de componentes conectadas : " << num_components << std::endl;
    //return num_components;
}

template<typename MemorySpace, typename MatrixType>
void coloring(const MatrixType& G)
{
    typedef typename MatrixType::index_type IndexType;
    typedef cusp::csr_matrix<IndexType,IndexType,MemorySpace> GraphType;

    GraphType G_csr(G);
    cusp::array1d<IndexType,MemorySpace> colors(G.num_rows, 0);

    //timer t;
    size_t max_color = cusp::graph::vertex_coloring(G_csr, colors);
    //std::cout << "Coloring time    : " << t.milliseconds_elapsed() << " (ms)." << std::endl;
    std::cout << "numero de colores necesarios : " << max_color << std::endl;

    // colores
    std::cout << "color de cada vertice" << std::endl;
    cusp::print(colors);


    std::cout << "cuantos vertices de cada color" << std::endl;
    if(max_color > 0)
    {
      cusp::array1d<IndexType,MemorySpace> color_counts(max_color);
      thrust::sort(colors.begin(), colors.end());
      thrust::reduce_by_key(colors.begin(),
                          colors.end(),
                          thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(),
                          color_counts.begin());
      cusp::print(color_counts);
    }

}


typedef cusp::csr_matrix<int,int,cusp::device_memory> csrmatrix;
void power5(csrmatrix &A)
{
  csrmatrix C(A);
  csrmatrix D(A);
  cusp::multiply(A, A, C); // C=A^2
  cusp::multiply(C, C, D); // D=A^4
  cusp::multiply(D, A, C); // C=A^5
  cusp::print(C);
}


int main()
{
  cusp::array2d<int,cusp::host_memory> A(9,9,0);
  make_adjacency_matrix(A);

  std::cout << "==== la matriz de adjacencia en formato denso ====" << std::endl;
  cusp::print(A);

  csrmatrix B(A);
  // la matriz
  std::cout << "==== la matriz de adjacencia en formato CSR ====" << std::endl;
  cusp::print(B);

  std::cout << "\n\n==== componentes conectadas ====" << std::endl;
  CC<cusp::device_memory>(B);

  std::cout << "\n\n==== colooooores ====" << std::endl;
  coloring<cusp::device_memory>(B);

  std::cout << "\n\n==== ¡caminos de 5 pasos entre dos nodos! ====" << std::endl;
  power5(B);

}

