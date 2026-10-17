# builds c2feat (training feature extractor) against a Codec2-mod source tree
C2MOD ?= ../../Codec2-mod
CFLAGS ?= -O2 -std=gnu11 -Wall -Wno-unused-function
SRC := $(wildcard $(C2MOD)/src/*.c)
all: c2feat
c2feat: c2feat.c $(SRC)
	$(CC) $(CFLAGS) -I$(C2MOD)/inc -o $@ $^ -lm
clean:
	rm -f c2feat
