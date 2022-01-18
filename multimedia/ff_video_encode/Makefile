
DEBUG        ?= 0
PRODUCTFORM  ?= soc

INSTALL_DIR    ?= release
top_dir :=../../..

ifeq ($(PRODUCTFORM),x86) # pcie mode
    CROSS_CC_PREFIX = x86_64-linux-
endif

ifeq ($(PRODUCTFORM),soc)
    CROSS_CC_PREFIX = aarch64-linux-gnu-
endif

ifeq ($(PRODUCTFORM),arm_pcie)
    CROSS_CC_PREFIX = aarch64-linux-gnu-
endif

ifeq ($(PRODUCTFORM),loongarch64)
    CROSS_CC_PREFIX = loongarch64-linux-gnu-
endif

CC  = $(CROSS_CC_PREFIX)gcc
CXX = $(CROSS_CC_PREFIX)g++

CPPFLAGS := -fPIC -Wall -Wl,--fatal-warning
ifeq ($(DEBUG), 0)
    CPPFLAGS += -O2
else
    CPPFLAGS += -g
endif

# NATIVE API SDK
NATIVE_SDK_HEADERS:=-I$(top_dir)/include/decode
NATIVE_SDK_LDFLAGS:=-L$(top_dir)/lib/decode/${PRODUCTFORM}
NATIVE_SDK_LDLIBS :=-lbmion -lbmjpulite -lbmjpuapi -lbmvpulite -lbmvpuapi -lbmvideo -lbmvppapi -lbmcv
#ifneq ($(PRODUCTFORM),soc) # for pcie x86_64 or pcie aarch64
NATIVE_SDK_HEADERS += -I$(top_dir)/include/bmlib

ifeq (${PRODUCTFORM}, x86)
NATIVE_SDK_LDFLAGS += -L$(top_dir)/lib/bmnn/pcie
else
NATIVE_SDK_LDFLAGS += -L$(top_dir)/lib/bmnn/$(PRODUCTFORM)
endif
NATIVE_SDK_LDLIBS  += -lbmlib


# FFMPEG SDK
FF_SDK_HEADERS := -I$(top_dir)/include/ffmpeg
FF_SDK_LDFLAGS := -L$(top_dir)/lib/ffmpeg/$(PRODUCTFORM)
FF_SDK_LDLIBS  := -lavfilter -lavformat -lavcodec -lswscale -lavutil -lswresample

CPPFLAGS += $(NATIVE_SDK_HEADERS) $(FF_SDK_HEADERS)
LDFLAGS  := $(NATIVE_SDK_LDFLAGS) $(FF_SDK_LDFLAGS)
LDLIBS   := $(FF_SDK_LDLIBS) $(NATIVE_SDK_LDLIBS) -lpthread -lstdc++

ifeq ($(PRODUCTFORM),loongarch64)
LDLIBS   += -lm -lpthread  -lresolv  -lrt  -ldl -lstdc++
endif

TARGET=test_ff_video_encode
MAKEFILE=Makefile
ALLOBJS=*.o
ALLDEPS=*.dep
RM=rm -rf
CP=cp -f

SOURCES := ff_video_encode.cpp

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

ff_video_encode.o : ff_video_encode.cpp $(MAKEFILE)
	$(CXX) $(CPPFLAGS) -c $< -o $@ -MD -MF $(@:.o=.dep)
