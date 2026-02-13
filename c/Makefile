CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

TARGET = graph_benchmark
SRCS = graph_benchmark.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

# Test with tiny network
test: $(TARGET)
	@if [ -f ../data/tiny_nodes.csv ]; then \
		./$(TARGET) ../data/tiny_nodes.csv ../data/tiny_edges.csv; \
	else \
		echo "Generate test data first: cd ../scripts && python generate_network.py --size tiny"; \
	fi

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: clean $(TARGET)
