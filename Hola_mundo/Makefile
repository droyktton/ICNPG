#
#
#	Compilador y flags del compilador y del linker
#
#
CC=gcc
CFLAGS=
CPPFLAGS=
LDFLAGS = 
#
#	Nombre del ejecutable
#
BIN=main
#
#	Archivos fuentes
#
SOURCES=$(shell echo *.c)
#
#	Objetos a partir de los fuentes
#
OBJECTS=$(patsubst %.c, %.o, $(SOURCES))
#
# 	Rules
#
all: $(BIN)

$(BIN): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ 

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ -c $<

.PHONY: all clean

clean: 
	rm -f .depend *.o $(BIN)
