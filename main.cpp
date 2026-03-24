#include "Histogram.h"
#include <iostream>

int main(void)
{
	Histogram histo;

	if (!histo.Initialize())
	{
		fprintf(stderr, "initialization failed \n");
		return 1;
	}
	histo.ProcessImage();
	histo.CloseBufHandles();

	return 0;
}