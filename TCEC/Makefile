TARGET1 := crypto
TARGET2 := image_measure
TARGET3 := communications
TARGET4 := operation_monitor
TARGET5 := dec


ccflags-y += -w

KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)
obj-m += $(TARGET1).o
obj-m += $(TARGET2).o  
obj-m += $(TARGET3).o
obj-m += $(TARGET4).o  
obj-m += $(TARGET5).o


default:
	make -C $(KDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean




install:
	insmod $(TARGET1).ko
	insmod $(TARGET2).ko
	insmod $(TARGET3).ko
	insmod $(TARGET4).ko

uninstall:
	rmmod $(TARGET4).ko
	rmmod $(TARGET3).ko
	rmmod $(TARGET2).ko
	rmmod $(TARGET1).ko

