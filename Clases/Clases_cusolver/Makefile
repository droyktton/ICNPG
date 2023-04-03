APPS=cusolver_demo_thrust cusolver_demo_cuda cusolver_mini_thrust cusolver_mini_cuda

all: ${APPS}

cusolver_demo_thrust: cusolver_demo_DnDsyevj_thrust.cu
	nvcc cusolver_demo_DnDsyevj_thrust.cu -lcusolver -o cusolver_demo_thrust

cusolver_demo_cuda: cusolver_demo_DnDsyevj.cu
	nvcc cusolver_demo_DnDsyevj.cu -lcusolver -o cusolver_demo_cuda

cusolver_mini_thrust: syevd_miniexample_thrust.cu
	nvcc syevd_miniexample_thrust.cu -lcusolver -o cusolver_mini_thrust

cusolver_mini_cuda: syevd_miniexample.cu
	nvcc syevd_miniexample.cu -lcusolver -o cusolver_mini_cuda


clean:
	rm -f ${APPS}

submit:	all
	qsub jobGPU; watch qstat
