
DEBUG        ?= 0
PRODUCTFORM  ?= soc

INSTALL_DIR    ?= release

top_dir :=../../..

ifeq ($(PRODUCTFORM),x86) # pcie mode
    CROSS_CC_PREFIX = x86_64-linux-
endif

ifeq ($(PRODUCTFORM),soc) # soc mode
    CROSS_CC_PREFIX = aarch64-linux-gnu-
endif

ifeq ($(PRODUCTFORM),arm_pcie) # pcie_arm64
    CROSS_CC_PREFIX = aarch64-linux-gnu-
endif

ifeq ($(PRODUCTFORM),loongarch64)
    CROSS_CC_PREFIX = loongarch64-linux-gnu-
endif

CC  = $(CROSS_CC_PREFIX)gcc
CXX = $(CROSS_CC_PREFIX)g++

CPPFLAGS := -fPIC -Wall -Wl,--fatal-warning -g -std=c++11
ifeq ($(DEBUG), 0)
    CPPFLAGS += -O2
else
    CPPFLAGS += -g
endif

ifneq ($(PRODUCTFORM),soc)
    CPPFLAGS += -DBM_PCIE_MODE
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

# For linking static libraries of ffmpeg
#FF_SDK_LDLIBS += -losip -lssl -lcrypto -lm -ldl -lpthread -lrt -lfreetype -lz -lresolv

CPPFLAGS += $(NATIVE_SDK_HEADERS) $(FF_SDK_HEADERS)
LDFLAGS  := $(NATIVE_SDK_LDFLAGS) $(FF_SDK_LDFLAGS)
LDLIBS   := $(FF_SDK_LDLIBS) $(NATIVE_SDK_LDLIBS) -lstdc++ -lpthread

ifeq ($(PRODUCTFORM),loongarch64)
LDLIBS   += -lm  -lresolv  -lrt  -ldl
endif

TARGET=test_ff_bmcv_transcode
MAKEFILE=Makefile
ALLOBJS=*.o
ALLDEPS=*.dep
RM=rm -rf
CP=cp -f

SOURCES := ff_avframe_convert.cpp ff_video_decode.cpp ff_video_encode.cpp test_ff_video_transcode.cpp

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

test_ff_video_transcode.o : test_ff_video_transcode.cpp $(MAKEFILE)
	$(CXX) $(CPPFLAGS) -c $< -o $@ -MD -MF $(@:.o=.dep)
