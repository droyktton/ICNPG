========================================================================
Los pixels de una imagen cuadrada de LxL en escala de grises cobraron vida y 
empezaron a desplazarse en direcciones aleatorias, ubicandose en nuevas coordenadas X, Y 
pero sin cambiar su valor de gris Z (ver figuras). 

Su misión:
Hacer un programa paralelo para volverlos a su lugar rapidamente, 
poder formar la imagen original a la que pertenecían, y poder comentársela a sus docentes.

* Dispone de los siguentes datos:

- Un fichero de tres columnas, X, Y, Z, donde X,Y son las nuevas coordenadas desplazadas 
de cada pixel de la imagen y Z su correpondiente valor de gris, que permanece inmutable 
en el desplazamiento del pixel. 

Las filas del fichero está ordenadas aleatoriamente. Es decir, el n-esimo pixel 
del fichero no es necesariamente el n-esimo pixel de la imagen original. 

- A partir de un generador de numeros aleatorios “philox” definido así
typedef r123::Philox2x32 philox; 
se sumaron desplazamientos aleatorios a la coordenada original de cada pixel x,y,z:

k[0]=z;  k[1]=0; c[0]=S; c[1]=0; 
r=philox(c,k);

newx=x+r[0]%L; newy=y+r[1]%L;
newz = z;

Donde S es la suma de todos los valores de z sobre todos los pixels.

Para poder ver la imagen deberá ordenar los pixels en el orden que aparecen en la imagen original, 
es decir cuando esta es guardada como vector, i.e. (x,y) -> x+y*L. 
Una vez hecho esto, el template generara un archivo PGM que Ud podra visualizar e interpretar.
=========================================================================


