#line 1 "Particles.cudafe1.gpu"
#line 4 "e:\\dropbox\\visualstudio\\chimera\\source\\chimera\\Particles.cuh"
struct EmitterData;
#line 1 "Particles.cudafe1.gpu"
typedef char __nv_bool;
#line 428 "C:/Program Files (x86)/Microsoft Visual Studio 11.0/VC/include\\crtdefs.h"
typedef unsigned __int64 size_t;
#include "crt/host_runtime.h"
#line 2 "e:\\dropbox\\visualstudio\\chimera\\source\\chimera\\Particles.cuh"
typedef unsigned UINT;
#line 4 "e:\\dropbox\\visualstudio\\chimera\\source\\chimera\\Particles.cuh"
struct EmitterData {
#line 6 "e:\\dropbox\\visualstudio\\chimera\\source\\chimera\\Particles.cuh"
float rand;
#line 7 "e:\\dropbox\\visualstudio\\chimera\\source\\chimera\\Particles.cuh"
float birthTime;
#line 8 "e:\\dropbox\\visualstudio\\chimera\\source\\chimera\\Particles.cuh"
float time;
#line 9 "e:\\dropbox\\visualstudio\\chimera\\source\\chimera\\Particles.cuh"
float tmp;};
#line 322 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
typedef struct float4 Plane;
#line 384 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
typedef struct float3 (*function)(struct float3, struct float3);
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);
#line 233 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
extern int fdividef();
#line 19 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__texture_type__ __text_var(ct_gradientTexture,::ct_gradientTexture);

#include "Particles.cudafe2.stub.c"
