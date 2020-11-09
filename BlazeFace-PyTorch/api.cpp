#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define RAWNN_API __declspec(dllexport)
#else
#define RAWNN_API 
#endif

#include "BlazeFace.h"

extern "C" 
{
	RAWNN_API void* facedetector_create();
	RAWNN_API void facedetector_destroy(void* ptr);
	RAWNN_API void facedetector_run(void* ptr, const unsigned char* rgb128);
	RAWNN_API int facedetector_num_faces(void* ptr);
	RAWNN_API void facedetector_get_results(void* ptr, float* result);
}

void* facedetector_create()
{
	return new F_BlazeFace;
}

void facedetector_destroy(void* ptr)
{
	delete (F_BlazeFace*)ptr;
}

void facedetector_run(void* ptr, const unsigned char* rgb128)
{
	((F_BlazeFace*)ptr)->Run(rgb128);
}

int facedetector_num_faces(void* ptr)
{
	const std::vector<Detection>& det = ((F_BlazeFace*)ptr)->GetResults();
	return (int)det.size();
}

void facedetector_get_results(void* ptr, float* result)
{
	const std::vector<Detection>& det = ((F_BlazeFace*)ptr)->GetResults();
	memcpy(result, det.data(), sizeof(Detection)*det.size());
}
