SITE_PACKAGE = --user

all:
	@echo "site-package: $(SITE_PACKAGE)"
	cd src; python setup.py install $(SITE_PACKAGE); cd ..

clean:
	cd src; python setup.py develop --uninstall; cd ..
	rm -rf src/build/ src/dist src/dct.egg-info/
	rm -f *.pyc
	rm -f src/*.pyc
