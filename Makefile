
compile:
	python setup.py build_ext --inplace
	rm -fr build

clean:
	rm -fr *.pyc

clobber: clean
	rm -frv out/*
	rm -fv *.c *.so

run: compile
	python generate_figures.py
