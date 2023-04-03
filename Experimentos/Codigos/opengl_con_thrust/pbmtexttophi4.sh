# primero copiar una imagen, fliperla con display, y convertirla a pgm
# convert -gravity center -extent 512x512 tref_flip.png -compress none tref_flip.pbm

# luego correr el script este
# ./pbmtexttophi4.scr tref_flip.pbm
# con eso ya se puede correr el programa phi4 y pedirle que carge la imagen

#para pbm
gawk '{if(NR>2){for(i=1;i<=NF;i++) printf("%d\n",$i)}}' $1 > $1.phi4

#para pgm
#gawk '{if(NR>3){for(i=1;i<=NF;i++) printf("%f\n",2*((255-$i)/255.-0.5))}}' $1 > $1.phi4
