#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "Particles.fatbin.c"
extern void __device_stub__Z15_computeEmitterP6float4S0_P6float3S2_P11EmitterDataS1_ffffj(struct float4 *, struct float4 *, struct float3 *, struct float3 *, struct EmitterData *, struct float3&, float, float, float, float, UINT);
extern void __device_stub__Z15_computeGravityP6float4P6float3f(struct float4 *, struct float3 *, float);
extern void __device_stub__Z18_computeTurbulenceP6float4P6float3S2_jj(struct float4 *, struct float3 *, struct float3 *, UINT, UINT);
extern void __device_stub__Z10_integrateP6float4P6float3S2_f(struct float4 *, struct float3 *, struct float3 *, float);
extern void __device_stub__Z21_computeGradientFieldP6float4P6float3S_(struct float4 *, struct float3 *, struct float4&);
extern void __device_stub__Z20_computeGravityFieldP6float4P6float3S_if(struct float4 *, struct float3 *, struct float4&, int, float);
extern void __device_stub__Z23_computeVelocityDampingP6float4P6float3f(struct float4 *, struct float3 *, float);
extern void __device_stub__Z13_computePlane6float4P6float3PS_(Plane&, struct float3 *, struct float4 *);
extern void __device_stub__Z12_reduce_max4P6float4P6float3(struct float4 *, struct float3 *);
extern void __device_stub__Z12_reduce_min4P6float4P6float3(struct float4 *, struct float3 *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_17_Particles_cpp1_ii_610dbc65(void);
#pragma section(".CRT$XCU",read)
__declspec(allocate(".CRT$XCU"))static void (*__dummy_static_init__sti____cudaRegisterAll_17_Particles_cpp1_ii_610dbc65[])(void) = {__sti____cudaRegisterAll_17_Particles_cpp1_ii_610dbc65};
void __device_stub__Z15_computeEmitterP6float4S0_P6float3S2_P11EmitterDataS1_ffffj(struct float4 *__par0, struct float4 *__par1, struct float3 *__par2, struct float3 *__par3, struct EmitterData *__par4, struct float3&__par5, float __par6, float __par7, float __par8, float __par9, UINT __par10){__cudaSetupArgSimple(__par0, 0Ui64);__cudaSetupArgSimple(__par1, 8Ui64);__cudaSetupArgSimple(__par2, 16Ui64);__cudaSetupArgSimple(__par3, 24Ui64);__cudaSetupArgSimple(__par4, 32Ui64);__cudaSetupArg(__par5, 40Ui64);__cudaSetupArgSimple(__par6, 52Ui64);__cudaSetupArgSimple(__par7, 56Ui64);__cudaSetupArgSimple(__par8, 60Ui64);__cudaSetupArgSimple(__par9, 64Ui64);__cudaSetupArgSimple(__par10, 68Ui64);__cudaLaunch(((char *)((void ( *)(struct float4 *, struct float4 *, struct float3 *, struct float3 *, struct EmitterData *, struct float3, float, float, float, float, UINT))_computeEmitter)));}
#line 39 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
void _computeEmitter( struct float4 *__cuda_0,struct float4 *__cuda_1,struct float3 *__cuda_2,struct float3 *__cuda_3,struct EmitterData *__cuda_4,struct float3 __cuda_5,float __cuda_6,float __cuda_7,float __cuda_8,float __cuda_9,UINT __cuda_10)
#line 40 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{__device_stub__Z15_computeEmitterP6float4S0_P6float3S2_P11EmitterDataS1_ffffj( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7,__cuda_8,__cuda_9,__cuda_10);
#line 119 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 1 "Particles.cudafe1.stub.c"
void __device_stub__Z15_computeGravityP6float4P6float3f( struct float4 *__par0,  struct float3 *__par1,  float __par2) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaLaunch(((char *)((void ( *)(struct float4 *, struct float3 *, float))_computeGravity))); }
#line 129 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
void _computeGravity( struct float4 *__cuda_0,struct float3 *__cuda_1,float __cuda_2)
#line 130 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{__device_stub__Z15_computeGravityP6float4P6float3f( __cuda_0,__cuda_1,__cuda_2);
#line 142 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 1 "Particles.cudafe1.stub.c"
void __device_stub__Z18_computeTurbulenceP6float4P6float3S2_jj( struct float4 *__par0,  struct float3 *__par1,  struct float3 *__par2,  UINT __par3,  UINT __par4) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaSetupArgSimple(__par3, 24Ui64); __cudaSetupArgSimple(__par4, 28Ui64); __cudaLaunch(((char *)((void ( *)(struct float4 *, struct float3 *, struct float3 *, UINT, UINT))_computeTurbulence))); }
#line 151 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
void _computeTurbulence( struct float4 *__cuda_0,struct float3 *__cuda_1,struct float3 *__cuda_2,UINT __cuda_3,UINT __cuda_4)
#line 152 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{__device_stub__Z18_computeTurbulenceP6float4P6float3S2_jj( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
#line 168 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 1 "Particles.cudafe1.stub.c"
void __device_stub__Z10_integrateP6float4P6float3S2_f( struct float4 *__par0,  struct float3 *__par1,  struct float3 *__par2,  float __par3) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaSetupArgSimple(__par3, 24Ui64); __cudaLaunch(((char *)((void ( *)(struct float4 *, struct float3 *, struct float3 *, float))_integrate))); }
#line 177 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
void _integrate( struct float4 *__cuda_0,struct float3 *__cuda_1,struct float3 *__cuda_2,float __cuda_3)
#line 178 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{__device_stub__Z10_integrateP6float4P6float3S2_f( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
#line 197 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 1 "Particles.cudafe1.stub.c"
void __device_stub__Z21_computeGradientFieldP6float4P6float3S_( struct float4 *__par0,  struct float3 *__par1,  struct float4&__par2) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArg(__par2, 16Ui64); __cudaLaunch(((char *)((void ( *)(struct float4 *, struct float3 *, struct float4))_computeGradientField))); }
#line 206 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
void _computeGradientField( struct float4 *__cuda_0,struct float3 *__cuda_1,struct float4 __cuda_2)
#line 207 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{__device_stub__Z21_computeGradientFieldP6float4P6float3S_( __cuda_0,__cuda_1,__cuda_2);
#line 243 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 1 "Particles.cudafe1.stub.c"
void __device_stub__Z20_computeGravityFieldP6float4P6float3S_if( struct float4 *__par0,  struct float3 *__par1,  struct float4&__par2,  int __par3,  float __par4) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArg(__par2, 16Ui64); __cudaSetupArgSimple(__par3, 32Ui64); __cudaSetupArgSimple(__par4, 36Ui64); __cudaLaunch(((char *)((void ( *)(struct float4 *, struct float3 *, struct float4, int, float))_computeGravityField))); }
#line 253 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
void _computeGravityField( struct float4 *__cuda_0,struct float3 *__cuda_1,struct float4 __cuda_2,int __cuda_3,float __cuda_4)
#line 254 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{__device_stub__Z20_computeGravityFieldP6float4P6float3S_if( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
#line 281 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 1 "Particles.cudafe1.stub.c"
void __device_stub__Z23_computeVelocityDampingP6float4P6float3f( struct float4 *__par0,  struct float3 *__par1,  float __par2) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaSetupArgSimple(__par2, 16Ui64); __cudaLaunch(((char *)((void ( *)(struct float4 *, struct float3 *, float))_computeVelocityDamping))); }
#line 291 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
void _computeVelocityDamping( struct float4 *__cuda_0,struct float3 *__cuda_1,float __cuda_2)
#line 292 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{__device_stub__Z23_computeVelocityDampingP6float4P6float3f( __cuda_0,__cuda_1,__cuda_2);
#line 301 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 1 "Particles.cudafe1.stub.c"
void __device_stub__Z13_computePlane6float4P6float3PS_( Plane&__par0,  struct float3 *__par1,  struct float4 *__par2) {  __cudaSetupArg(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 16Ui64); __cudaSetupArgSimple(__par2, 24Ui64); __cudaLaunch(((char *)((void ( *)(Plane, struct float3 *, struct float4 *))_computePlane))); }
#line 334 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
void _computePlane( Plane __cuda_0,struct float3 *__cuda_1,struct float4 *__cuda_2)
#line 335 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{__device_stub__Z13_computePlane6float4P6float3PS_( __cuda_0,__cuda_1,__cuda_2);
#line 351 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 1 "Particles.cudafe1.stub.c"
void __device_stub__Z12_reduce_max4P6float4P6float3( struct float4 *__par0,  struct float3 *__par1) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaLaunch(((char *)((void ( *)(struct float4 *, struct float3 *))_reduce_max4))); }
#line 386 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
void _reduce_max4( struct float4 *__cuda_0,struct float3 *__cuda_1)
#line 387 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{__device_stub__Z12_reduce_max4P6float4P6float3( __cuda_0,__cuda_1);

}
#line 1 "Particles.cudafe1.stub.c"
void __device_stub__Z12_reduce_min4P6float4P6float3( struct float4 *__par0,  struct float3 *__par1) {  __cudaSetupArgSimple(__par0, 0Ui64); __cudaSetupArgSimple(__par1, 8Ui64); __cudaLaunch(((char *)((void ( *)(struct float4 *, struct float3 *))_reduce_min4))); }
#line 391 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
void _reduce_min4( struct float4 *__cuda_0,struct float3 *__cuda_1)
#line 392 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{__device_stub__Z12_reduce_min4P6float4P6float3( __cuda_0,__cuda_1);

}
#line 1 "Particles.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T27) {  __nv_dummy_param_ref(__T27); __cudaRegisterEntry(__T27, ((void ( *)(struct float4 *, struct float3 *))_reduce_min4), _reduce_min4, (-1)); __cudaRegisterEntry(__T27, ((void ( *)(struct float4 *, struct float3 *))_reduce_max4), _reduce_max4, (-1)); __cudaRegisterEntry(__T27, ((void ( *)(Plane, struct float3 *, struct float4 *))_computePlane), _computePlane, (-1)); __cudaRegisterEntry(__T27, ((void ( *)(struct float4 *, struct float3 *, float))_computeVelocityDamping), _computeVelocityDamping, (-1)); __cudaRegisterEntry(__T27, ((void ( *)(struct float4 *, struct float3 *, struct float4, int, float))_computeGravityField), _computeGravityField, (-1)); __cudaRegisterEntry(__T27, ((void ( *)(struct float4 *, struct float3 *, struct float4))_computeGradientField), _computeGradientField, (-1)); __cudaRegisterEntry(__T27, ((void ( *)(struct float4 *, struct float3 *, struct float3 *, float))_integrate), _integrate, (-1)); __cudaRegisterEntry(__T27, ((void ( *)(struct float4 *, struct float3 *, struct float3 *, UINT, UINT))_computeTurbulence), _computeTurbulence, (-1)); __cudaRegisterEntry(__T27, ((void ( *)(struct float4 *, struct float3 *, float))_computeGravity), _computeGravity, (-1)); __cudaRegisterEntry(__T27, ((void ( *)(struct float4 *, struct float4 *, struct float3 *, struct float3 *, struct EmitterData *, struct float3, float, float, float, float, UINT))_computeEmitter), _computeEmitter, (-1)); __cudaRegisterGlobalTexture(__T27, __text_var(ct_gradientTexture,::ct_gradientTexture), 3, 0, 0); }
static void __sti____cudaRegisterAll_17_Particles_cpp1_ii_610dbc65(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }
