INT8_MODE ?=0

VPATH=./src:./examples

INCLUDE =  -Iinclude \
	    -I${REL_TOP}/include \
		-I${REL_TOP}/include/third_party/boost/include \
		-I${REL_TOP}/include/third_party/glog/include \
		-I${REL_TOP}/include/third_party/gflags/include \
		-I${REL_TOP}/include/third_party/protobuf/include

LIB_PATH = -L .\
	    -L${REL_TOP}/lib \
	    -L${REL_TOP}/lib/bmcompiler \
	    -L${REL_TOP}/lib/calibration-tools \
	    -L${REL_TOP}/lib/thirdparty/x86



CXX = g++
CXXFLAGS = -g  -Wno-sign-compare -Wall -DUSE_OPENCV -fopenmp -std=c++11  -pthread -fPIC  ${INCLUDE}

CXXFLAGS += -DUSE_UFW=1 -DCPU_ONLY=1 


ifeq ($(INT8_MODE), 1)
CXXFLAGS += -DINT8_MODE 
endif
CXXFLAGS += -DINT8_COEFF_FUNC
  

LIBS = -law_face -lglog -lprotobuf -lpthread -lboost_system -lufw -pthread 
LIBS += $(shell pkg-config --libs opencv)



BIN_DIR = bin/
TARGET_LIB = libaw_face.so
TARGET_BIN = $(BIN_DIR)demo_detect

OBJ_DIR = obj/
EXEC_DIR = $(OBJ_DIR)examples/
OBJS = face_detect.o  face_util.o 
LIB_OBJS = $(addprefix ${OBJ_DIR}, ${OBJS})
TARGET_OBJ = $(addprefix $(EXEC_DIR), demo_detect.o)


all: target $(TARGET_LIB) $(TARGET_BIN) 

target:
	mkdir -p ${BIN_DIR} ${OBJ_DIR} ${EXEC_DIR}

$(OBJ_DIR)%.o: %.cpp
	$(CXX) -c $< -o $@ -fPIC -DBUILD_LIB $(CXXFLAGS)

$(EXEC_DIR)%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(TARGET_LIB): $(LIB_OBJS)
	$(CXX) -fPIC  -shared -o $@ $^ $(CXXFLAGS)

$(TARGET_BIN): $(TARGET_OBJ)
	  $(CXX) -fPIC   $^ -o $@ $(LIB_PATH) $(LIBS)



clean:
	-rm -f *.o *.so *.a
	-rm -rf ${BIN_DIR} ${OBJ_DIR}
	-rm -rf models/squeezenet/*test_*

clean_all:
	-rm -f *.o *.so *.a
	-rm -rf ${BIN_DIR} ${OBJ_DIR}
	-rm -rf models/squeezenet/*_unique_top*
