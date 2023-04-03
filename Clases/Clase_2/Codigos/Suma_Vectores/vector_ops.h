#pragma once

/***********************************************/
/* Librería que implementa operaciones entre   */
/* vectores                                    */
/***********************************************/

      
int vector_suma_sec(float *v1, float *v2, int dim);
int vector_suma_par(float *v1, float *v2, int dim);
int vector_iguales(float *v1, float *v2, int dim);

int vector_inicializacion_random(float *v, int dim);
int vector_imprimir(float *v, int dim);
int vector_inicializacion_unos(float *v, int dim);
