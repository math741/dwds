CFLAGS ?= -O3 -march=native -fopenmp -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wno-unused-parameter
SRC = mlg_core.c mlg_know.c mlg_router.c mlg_text.c

mlg_cli: $(SRC) mlg_know_build.c mlg_cli.c mlg.h
	$(CC) $(CFLAGS) -o $@ $(SRC) mlg_know_build.c mlg_cli.c -lm

clean:
	rm -f mlg_cli
