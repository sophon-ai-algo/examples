##############################################################################
#
#    Copyright (c) 2016-2022 by Bitmain Technologies Inc. All rights reserved.
#
#    The material in this file is confidential and contains trade secrets
#    of Bitmain Technologies Inc. This is proprietary information owned by
#    Bitmain Technologies Inc. No part of this work may be disclosed,
#    reproduced, copied, transmitted, or used in any way for any purpose,
#    without the express written permission of Bitmain Technologies Inc.
#
##############################################################################

LIBRARIES = bmrt bmlib pthread boost_system
TARGET_BIN = classify
ROOT_DIR = $(REL_TOP)

CXX = g++
CXXFLAGS += -I. -I$(ROOT_DIR)/include/bmruntime -I$(ROOT_DIR)/include/bmlib
CXXFLAGS += -Wall -DUSE_OPENCV -std=c++11
CXXFLAGS += -march=native
CXXFLAGS += -DTRANS=0
CXXFLAGS += -O0 -g3 -DUSE_MULTICARDS=1

LDFLAGS += $(shell pkg-config --libs opencv)
LDFLAGS += -L. -L$(ROOT_DIR)/lib/bmnn/pcie
LDFLAGS += $(foreach library,$(LIBRARIES),-l$(library))


all: $(TARGET_BIN)

$(TARGET_BIN):%:%.o
	$(CXX) $^ -o $@ $(LDFLAGS) -pthread

clean:
	-rm -f *.o $(TARGET_LIB) *.a
	-rm -f $(TARGET_BIN)
