set cbrange [-1:1]; 
set tit sprintf("%d",i)
plot [0:L-1][0:L-1] 'evolucion.dat' index i matrix with image t ''
#pause 1
i=i+1
print i
if(i<iend) reread
