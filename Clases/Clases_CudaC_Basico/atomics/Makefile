APPS=histo reduce histothrust

all: ${APPS}

histo:	histograma.cu
	nvcc histograma.cu -lnvToolsExt -o histo

reduce: reduccion.cu
	nvcc reduccion.cu -lnvToolsExt -o reduce

histothrust: histograma_thrust_sin_atomics.cu
	nvcc histograma_thrust_sin_atomics.cu  -lnvToolsExt -o histothrust

clean:
	rm -f ${APPS}






