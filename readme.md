# ml-example-knn

### Dependencies

- libpython2.7-dev
- opencv4

### Build

##### Linux

```console
user@workstation:<some-dir>$ git clone --recurse https://github.com/ramirsultanov/ml-example-knn.git
user@workstation:<some-dir>$ g++ -std=c++17 main.cxx -I/usr/lib/python2.7/Python.h -lpython2.7 `pkg-config --cflags --libs opencv4`
```

### Run

##### Linux

```console
user@workstation:<some-dir>$ ./a.out
```

