# Makefile example

CFLAGS=`pkg-config qibolab --cflags`
LIBS=`pkg-config qibolab --libs`

all: example

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf example
