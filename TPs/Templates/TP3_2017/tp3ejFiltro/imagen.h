

/* entrada y salida de imagenes */
void leer_imagen(const char * nombre, float *imagen, int filas, int cols);

void salvar_imagen(const char * nombre, float *imagen, int filas, int cols);

/* secuencial */
void filtro_sec_promedio(float *imagen_in, float *imagen_out, int filas, int cols, float *filtro);

void filtro_sec_enfocado(float *imagen_in, float *imagen_out, int filas, int cols, float *filtro);

void inicializar_filtro_promedio(float *filtro, int tamFiltro);

/* paralelo */
void filtro_par_promedio(float *d_imagen_in, float *d_imagen_out, int filas, int cols, float *d_filtro);


