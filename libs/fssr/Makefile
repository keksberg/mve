MVE_ROOT := ../..
TARGET := libmve_fssr.a
include ${MVE_ROOT}/Makefile.inc

CXXFLAGS += -fPIC -I${MVE_ROOT}/libs ${OPENMP}
LDLIBS += -lpng -ltiff -ljpeg ${OPENMP}

SOURCES := $(wildcard [^_]*.cc) $(wildcard [^_]*.cu)
${TARGET}: ${SOURCES:.cc=.o} ${SOURCES:.cu=.o}
	$(AR) rcs $@ $^

%.o: %.cu
	nvcc -c $^ -o $@

clean:
	${RM} $(TARGET) *.o Makefile.dep

.PHONY: clean
