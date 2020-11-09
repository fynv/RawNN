#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define RAWNN_API __declspec(dllexport)
#else
#define RAWNN_API 
#endif

#include "Xception.h"


extern "C"
{
	RAWNN_API void* classifier_create();
	RAWNN_API void classifier_destroy(void* ptr);
	RAWNN_API float classifier_run(void* ptr, const unsigned char* rgb128);	
}


void* classifier_create()
{
	return new Xception;
}

void classifier_destroy(void* ptr)
{
	delete (Xception*)ptr;
}

float classifier_run(void* ptr, const unsigned char* rgb128)
{
	return ((Xception*)ptr)->Run(rgb128);
}

