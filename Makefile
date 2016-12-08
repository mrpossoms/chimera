OUT=chimera

CXXC=g++
CXX_FLAGS=-g -ggdb -std=c++11

EXT=./external

# use the findingds of the findlibs.sh script
include findings.mk

SRC=./src
OBJ=./obj

SRC_VISUAL=$(SRC)/visual
SRC_GL=$(SRC_VISUAL)/GL_1_3

# inter-project includes
INC_PATHS+=-I$(EXT)/linmath.h -I$(EXT)/opt.h -I$(SRC) -I$(SRC_VISUAL)

GLFW_LINK=-lglfw3 -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo -lpng
LINK=$(GLFW_LINK)

all: findings.mk
	$(CXXC) $(CXX_FLAGS) $(INC_PATHS) $(LIB_PATHS) $(SRC)/main.cpp -o $(OUT) $(LINK) $(OBJ)/*.o

task.o:
	$(CXXC) $(CXX_FLAGS) $(INC_PATHS) -c $(SRC)/task.cpp -o $(OBJ)/task.o

.PHONY: visual.gl
visual.gl: material.gl.o viewer.gl.o viewer.o percept.o noise.o task.o

material.gl.o:
	$(CXXC) $(CXX_FLAGS) $(INC_PATHS) -c $(SRC_GL)/material.gl.cpp -o $(OBJ)/material.gl.o

viewer.gl.o:
	$(CXXC) $(CXX_FLAGS) $(INC_PATHS) -c $(SRC_GL)/viewer.gl.cpp -o $(OBJ)/viewer.gl.o

percept.o:
	$(CXXC) $(CXX_FLAGS) $(INC_PATHS) -c $(SRC)/percept.cpp -o $(OBJ)/percept.o

viewer.o:
	$(CXXC) $(CXX_FLAGS) $(INC_PATHS) -c $(SRC_VISUAL)/viewer.cpp -o $(OBJ)/viewer.o

noise.o:
	$(CXXC) $(CXX_FLAGS) $(INC_PATHS) -c $(SRC_VISUAL)/noise.cpp -o $(OBJ)/noise.o

# material.gl.o:
# 	$(CXXC) $(CXX_FLAGS) $(INC_PATHS) $(LIB_PATHS)

clear-data:
	rm ./data/*

clean: clear-data
	rm $(OBJ)/*.o || true
	rm $(OUT) || true

findings.mk:
	./findlibs.sh
	make -C . all
