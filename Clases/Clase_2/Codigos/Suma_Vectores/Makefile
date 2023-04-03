CC=nvcc
CFLAGS=-arch=sm_21

BIN=main
SOURCES=$(shell echo *.cu)
OBJECTS=$(patsubst %.cu, %.o, $(SOURCES))
LDFLAGS = 

# rules

all: $(BIN)

$(BIN): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ 

%.o: %.cu
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ -c $<

-include .depend

.depend: $(SOURCES) *.h
	$(CC) -M $(CPPFLAGS) $(SOURCES) > .depend

.PHONY: all clean

clean: 
	rm -f .depend *.o $(BIN)
