HIPCC      ?= hipcc
HIPFLAGS   ?= -O3 -std=c++17
ARCHFLAG   ?= --offload-arch=gfx942

BUILDDIR   := build
INCDIR     := include
BENCHDIR   := benchmarks

SRCS       := $(wildcard $(BENCHDIR)/*.hip)
TARGETS    := $(patsubst $(BENCHDIR)/%.hip,$(BUILDDIR)/%,$(SRCS))

# Per-target link flags (e.g. gemm_roofline needs hipBLASLt)
LDFLAGS_gemm_roofline = -lhipblaslt

.PHONY: all clean list

all: $(TARGETS)

$(BUILDDIR)/%: $(BENCHDIR)/%.hip $(wildcard $(INCDIR)/*.h) | $(BUILDDIR)
	$(HIPCC) $(HIPFLAGS) $(ARCHFLAG) -I$(INCDIR) -o $@ $< $(LDFLAGS_$*)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)

list:
	@echo "Benchmarks:"
	@for t in $(TARGETS); do echo "  $$t"; done
