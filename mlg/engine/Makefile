CFLAGS ?= -O3 -march=native -D_DEFAULT_SOURCE -fopenmp -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wno-unused-parameter -Wno-format-truncation
SRC = mlg_core.c mlg_know.c mlg_router.c mlg_text.c mlg_mind.c

mlg_cli: $(SRC) mlg_know_build.c mlg_cli.c mlg.h
	$(CC) $(CFLAGS) -o $@ $(SRC) mlg_know_build.c mlg_cli.c -lm

# the lab drives the same mind through ctypes (mlgsub/mind.py)
libmlg.so: $(SRC) mlg_know_build.c mlg.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(SRC) mlg_know_build.c -lm

all: mlg_cli libmlg.so

clean:
	rm -f mlg_cli libmlg.so
