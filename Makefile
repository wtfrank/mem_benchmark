# Compiler settings
CC = gcc
CFLAGS = -O3 -Wall -Wextra -mavx -march=native -g
# CFLAGS = -Wall -Wextra -mavx -march=native -g
LDFLAGS = -lpthread

# Target executable name
TARGET = mem_benchmark

# Source files
SRC = benchmark.c

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Clean up build files
clean:
	rm -f $(TARGET)

# Run the benchmark (requires enough free RAM)
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
