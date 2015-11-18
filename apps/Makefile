APPS = $(wildcard */)

default: all

qmake:
	cd umve && qmake

all links: $(APPS)
	for app in $^; do \
		dest=$(HOME)/bin/$${app%/}; \
		rm -f $$dest; \
		ln -sr $$app$${app%/} $$dest; \
	done

all clean: $(APPS)

$(APPS): qmake
ifneq ($(MAKECMDGOALS), links)
	$(MAKE) -C $@ $(MAKECMDGOALS)
endif

.PHONY: all links clean
