minz=-0.7
maxz=0.01
set cbrange [minz:maxz]
splot [0:l][0:l][minz:maxz] 'peli.dat' matrix index i w pm3d tit sprintf("tiempo=%d",i)

# interactivo, usar flechitas para avanzar o retroceder
# gnuplot> l=15;i=0;load 'movie.gnu'
#pause mouse key
#print MOUSE_KEY
#if(MOUSE_KEY==1011 && i>0) i=i-1
#if(MOUSE_KEY==1009  && i<500) i=i+1

# o bien hacer avanzar todo para generar un gif animado
# gnuplot> set term gif animate; set out 'peli.gif';l=15;i=0;load 'movie.gnu'
print i
i=i+1

if(i<2500) reread
