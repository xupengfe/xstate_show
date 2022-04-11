BIN = xstate

all: $(BIN)

xstate:
	$(CC) -m64 -o $(BIN) -O2 -g -std=gnu99 -pthread -Wall -no-pie -mno-sse -mno-mmx -mno-sse2 -mno-avx -mno-pku -DCAN_BUILD_32 -DCAN_BUILD_64 xstate.c helpers.h -lrt -ldl

clean:
	rm -f $(BIN)
 
