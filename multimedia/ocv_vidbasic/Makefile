
DEBUG        ?= 0
PRODUCTFORM  ?= soc
BM_MEDIA_ION ?= 0

INSTALL_DIR    ?= release
top_dir :=../../..


ifeq ($(PRODUCTFORM),x86) # pcie mode
    CROSS_CC_PREFIX = x86_64-linux-
else # pcie_arm64 and soc mode
    CROSS_CC_PREFIX = aarch64-linux-gnu-
endif

CC  = $(CROSS_CC_PREFIX)gcc
CXX = $(CROSS_CC_PREFIX)g++

CPPFLAGS := -std=c++11 -fPIC -Wall -Wl,--fatal-warning
ifeq ($(DEBUG), 0)
    CPPFLAGS += -O2
else
    CPPFLAGS += -g
endif

# NATIVE API SDK
NATIVE_SDK_HEADERS:=-I$(top_dir)/include/decode
NATIVE_SDK_LDFLAGS:=-L$(top_dir)/lib/decode/${PRODUCTFORM}
NATIVE_SDK_LDLIBS :=-lbmion -lbmjpulite -lbmjpuapi -lbmvpulite -lbmvpuapi -lbmvideo -lbmvppapi -lyuv

# FFMPEG SDK
FF_SDK_HEADERS := -I$(top_dir)/include/ffmpeg
FF_SDK_LDFLAGS := -L$(top_dir)/lib/ffmpeg/$(PRODUCTFORM)
FF_SDK_LDLIBS  := -lavcodec -lavformat -lavutil -lswresample -lswscale

# OpenCV SDK
OCV_SDK_HEADERS := -I$(top_dir)/include/opencv/opencv4
OCV_SDK_LDFLAGS := -L$(top_dir)/lib/opencv/$(PRODUCTFORM)
OCV_SDK_LDLIBS  := -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio

# BMCV SDK
BMCV_SDK_HEADERS := -I$(top_dir)/include/bmlib
BMCV_SDK_LDFLAGS := -L$(top_dir)/lib/bmnn/$(PRODUCTFORM)
ifeq (${PRODUCTFORM}, x86)
BMCV_SDK_LDFLAGS :=  -L$(top_dir)/lib/bmnn/pcie
endif
BMCV_SDK_LDLIBS  := -lbmcv -lbmlib

CPPFLAGS += $(NATIVE_SDK_HEADERS) $(FF_SDK_HEADERS) $(OCV_SDK_HEADERS) $(BMCV_SDK_HEADERS)
LDFLAGS  := $(NATIVE_SDK_LDFLAGS) $(FF_SDK_LDFLAGS) $(OCV_SDK_LDFLAGS) $(BMCV_SDK_LDFLAGS)
LDLIBS   := $(NATIVE_SDK_LDLIBS) $(FF_SDK_LDLIBS) $(OCV_SDK_LDLIBS) $(BMCV_SDK_LDLIBS) -lstdc++


TARGET=test_ocv_vidbasic
MAKEFILE=Makefile
ALLOBJS=*.o
ALLDEPS=*.dep
RM=rm -rf
CP=cp -f

SOURCES := vidbasic.cpp

OBJECTPATHS:=$(patsubst %.cpp,%.o,$(SOURCES))

.phony: all clean

all: $(TARGET)

$(TARGET): $(OBJECTPATHS)
	$(CC) -o $@ $(OBJECTPATHS) $(LDFLAGS) $(LDLIBS)

install: $(TARGET)
	install -d $(INSTALL_DIR)/bin
	install $(TARGET) $(INSTALL_DIR)/bin

uninstall:
	$(RM) $(INSTALL_DIR)/bin/$(TARGET) 

clean:
	$(RM) $(TARGET)
	$(RM) $(ALLDEPS)
	$(RM) $(ALLOBJS)

ocv_video_xcode.o : ocv_video_xcode.cpp $(MAKEFILE)
	$(CXX) $(CPPFLAGS) -c $< -o $@ -MD -MF $(@:.o=.dep)
