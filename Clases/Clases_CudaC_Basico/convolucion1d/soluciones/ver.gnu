set key left ; set yla 'tiempo [ms]'; set xla 'M'; set title 'N=8388608'; plot 'tiempos.txt' u 1:3 w lp t 'cpu', '' u 1:4 t 'gpu' w lp, '' u 1:5 t 'gpu con copia' w lp, x*10 t '~x'
