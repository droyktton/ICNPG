#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "hola_mundo_io.h"

#ifndef VECES
#define VECES 10
#endif

int main()
{
    int i;
    for(i = 0; i < VECES; i++)
    {   
        printf("%d: ",i);
        imprime_hola_mundo();
    }
    printf(" Hola Flavioc\n");
    printf(" Hola koltona\n");
    printf(" Hola Karina\n");
    printf(" Hola Moni\n");
    return 0;
}

