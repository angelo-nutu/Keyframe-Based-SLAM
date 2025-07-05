.PHONY: all dir debug release clean start

all: debug

dir:
	@mkdir -p build 2>/dev/null

debug: dir
	@cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j$$(nproc)

release: dir
	@cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$$(nproc)

clean:
	@rm -rf build

start:
	@./build/vo
