#N=16;
system(sprintf("./a.out %d",N)); 
set multi lay 2,3; 
do for[file in "A.dat B.dat Ccuda.dat Ccpu.dat Ccublas.dat Ccublasxt.dat"]{
	set tit file; plot file matrix w image
}; 
unset multi
