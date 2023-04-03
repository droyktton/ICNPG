/*
Compilacion (en susa):
nvcc -arch=sm_20 -o pbo_opengl_template pbo_opengl_template.cu -DSMOOTHDISORDER -lglut -lGLU -lGL -lcufft

Este file contiene codigo exclusivo a opengl. 
Para conectar con un sistema simulado, se agregan 
misistema.h => clase del sistema simulado. TOTALMENTE INDEPENDIENTE de opengl
interface.h => funciones utiles para mapear el sistema simulado al buffer de opengl. DEPENDIENTE DE OPENGL.
*/
#include "misistema.h"
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <GL/gl.h> 
#include <GL/glut.h> 
#include <cuda_gl_interop.h> 
#include <GL/glext.h> 
#include <GL/glx.h> 
#define GET_PROC_ADDRESS( str ) glXGetProcAddress( (const GLubyte *)str ) 

static void HandleError( cudaError_t err, const char *file,  int line ) { 
    if (err != cudaSuccess) { 
            printf( "%s in %s at line %d\n", cudaGetErrorString( err ),  file, line ); 
            exit( EXIT_FAILURE ); 
    } 
} 
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 


PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL; 
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL; 
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL; 
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL; 

#define     DIM    512
#define REFRESH_DELAY     10 //ms

GLuint  bufferObj; 
cudaGraphicsResource *resource; 

// define la interface entre opengl y mi sistema //////
#include "interface.h"                                //
////////////////////////////////////////////////////////

static void draw_func( void ) { 
  rescale();
  change_pixels();
  glDrawPixels( DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0 ); 
  glutSwapBuffers(); 
}

static void key_func( unsigned char key, int x, int y ) { 
  switch (key) { 
    case 27: 
        HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) ); 
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 ); 
        glDeleteBuffers( 1, &bufferObj ); 
        exit(0); 
        break;
    case 32: // space
        paso=0; 
        break;
    case 8: // backspace
        Sptr->reinicializar();
        break;
    case 9: // tab 
        rescale();
        break;
    case 49:
        paso=1;
        break;
    case '0':
        paso=0;
        break;
    case 50:
        paso=10;
        break;
    case 51:
        paso=100;
        break;
    case 52:
        paso=1000;
        break;
    case 53:
        paso=10000;
        break;
    case 54:
        paso=100000;
        break;
    case 'C':
        Sptr->addcircle(DIM*0.5, DIM*0.25, DIM*0.12,1);
        Sptr->addcircle(DIM*0.5, DIM*0.75, DIM*0.12,1);
        break;
    case 'c':
        Sptr->addcircle(DIM*0.5, DIM*0.5, DIM*0.12,1);
        break;
    case 't':
        Sptr->addtext(0,0,1,512*512);
        break;
    case 'w':
        Sptr->detect_wall(outwall);
        break;
    case 'x':
        signo*=-1.0;
        break;    
    case 'h':
        hext+=0.005;
	Sptr->set_hext(hext);        
	std::cout << "hext=" << hext << std::endl;
	break;    
    case 'H':
        hext-=0.005;
	Sptr->set_hext(hext);        
	std::cout << "hext=" << hext << std::endl;
        break;    
    default:
        break;
  } 
} 

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    sqx=(x>0 && x<DIM)?(x):(sqx); sqy=(y>0 && y<DIM)?(y):(sqy);
    std::cout << sqx << " (clicks) " << sqy << " " << signo << std::endl;
    //Sptr->addsquare(sqx,DIM-sqy, 50);
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    sqx=(x>0 && x<DIM)?(x):(sqx); sqy=(y>0 && y<DIM)?(y):(sqy);

    //std::cout << sqx << " (motion) " << sqy << std::endl;
    Sptr->addsquare(sqx,DIM-sqy, 10, signo);

}






void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

int main(int argc, char *argv[]) { 

  //int numero;
  //std::cin >> numero;
  //test(numero,DIM);
  //exit(1);

  ////// declarar/inicializar sistema //////
  sistema S(DIM); Sptr=&S;
  inicializar_variables_globales_sistema();
  //////////////////////////////////////////

  cudaGLSetGLDevice( 0 ); 

  glutInit( &argc, argv ); 
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA ); 
  glutInitWindowSize( DIM, DIM ); 
  glutCreateWindow( "Mi peliculita" ); 

  glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer"); 
  glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers"); 
  glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers"); 
  glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData"); 

  glGenBuffers( 1, &bufferObj ); 
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj ); 
  glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB ); 


  cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone ); 

// set up GLUT and kick off main loop 
  glutKeyboardFunc( key_func ); 
  glutDisplayFunc( draw_func ); 
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  glutTimerFunc(REFRESH_DELAY, timerEvent,0);
  glutMainLoop(); 
}
