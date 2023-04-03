#!/usr/bin/gnuplot -persist

# dibuja todas
#plot [0:] for[ind=0:999] './histogramas.dat' index ind u 1:2 w l t '', exp(-(sin(2 *pi* x) + 0.25 *sin(4*pi* x))/0.75)*0.64

# animacion
ind=0
do for[aux=0:1000000]{ plot [0:]  './histogramas.dat' index ind u 1:2 w l t sprintf("%d",ind), exp(-(sin(2 *pi* x) + 0.25 *sin(4*pi* x))/0.75)*0.64; pause mouse key; if(MOUSE_KEY==97){ind=(ind+1<1000)?(ind+1):(1000)}; if(MOUSE_KEY==122){ind=(ind-1>0)?(ind-1):(0)}; reread}

#    EOF
