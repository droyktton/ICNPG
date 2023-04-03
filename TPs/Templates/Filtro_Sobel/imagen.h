

/* entrada y salida de imagenes */
void leer_imagen(const char * nombre, float *imagen, int filas, int cols);
void salvar_imagen(const char * nombre, float *imagen, int filas, int cols);

void filtro_sec(float *imagen_in, float *imagen_out, int filas, int cols, float *filtro);

void inicializar_filtro(float *filtro, int tamFiltro);

void inicializar_filtro_sobel_horizontal(float *filtro, int tamFiltro);

void inicializar_filtro_sobel_vertical(float *filtro, int tamFiltro);

void aplicar_filtro_sobel_sec(float *imagen_in, float *imagen_out, int filas, int cols);

void calcular_g(float *g_x, float *g_y, float *imagen_out, int filas, int cols);

/* paralelo */
void filtro_par(float *d_imagen_in, float *d_imagen_out, int filas, int cols, float *d_filtro);

void aplicar_filtro_sobel_par(float *d_imagen_in, float *d_imagen_out, int filas, int cols);