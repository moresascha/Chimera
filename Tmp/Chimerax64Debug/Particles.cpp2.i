#line 1 "E:/Dropbox/VisualStudio/Chimera/Source/../Tmp/Chimerax64Debug//Particles.cudafe1.gpu"
typedef char __nv_bool;
#line 672 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"
struct cudaArray;
#line 1298 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"
struct CUstream_st;
#line 74 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\cuda_texture_types.h"
struct _Z7textureI6float4Li3EL19cudaTextureReadMode0EE;
#line 198 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUipcMem_flags_enum {
#line 199 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1};
#line 207 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUctx_flags_enum {
#line 208 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CTX_SCHED_AUTO,
#line 209 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CTX_SCHED_SPIN,
#line 210 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CTX_SCHED_YIELD,
#line 211 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CTX_SCHED_BLOCKING_SYNC = 4,
#line 212 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CTX_BLOCKING_SYNC = 4,
#line 215 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CTX_SCHED_MASK = 7,
#line 216 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CTX_MAP_HOST,
#line 217 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CTX_LMEM_RESIZE_TO_MAX = 16,
#line 218 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CTX_FLAGS_MASK = 31};
#line 224 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUstream_flags_enum {
#line 225 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_STREAM_DEFAULT,
#line 226 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_STREAM_NON_BLOCKING};
#line 232 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUevent_flags_enum {
#line 233 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_EVENT_DEFAULT,
#line 234 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_EVENT_BLOCKING_SYNC,
#line 235 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_EVENT_DISABLE_TIMING,
#line 236 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_EVENT_INTERPROCESS = 4};
#line 242 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUarray_format_enum {
#line 243 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_AD_FORMAT_UNSIGNED_INT8 = 1,
#line 244 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_AD_FORMAT_UNSIGNED_INT16,
#line 245 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_AD_FORMAT_UNSIGNED_INT32,
#line 246 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_AD_FORMAT_SIGNED_INT8 = 8,
#line 247 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_AD_FORMAT_SIGNED_INT16,
#line 248 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_AD_FORMAT_SIGNED_INT32,
#line 249 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_AD_FORMAT_HALF = 16,
#line 250 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_AD_FORMAT_FLOAT = 32};
#line 256 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUaddress_mode_enum {
#line 257 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TR_ADDRESS_MODE_WRAP,
#line 258 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TR_ADDRESS_MODE_CLAMP,
#line 259 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TR_ADDRESS_MODE_MIRROR,
#line 260 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TR_ADDRESS_MODE_BORDER};
#line 266 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUfilter_mode_enum {
#line 267 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TR_FILTER_MODE_POINT,
#line 268 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TR_FILTER_MODE_LINEAR};
#line 274 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUdevice_attribute_enum {
#line 275 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
#line 276 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
#line 277 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
#line 278 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
#line 279 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
#line 280 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
#line 281 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
#line 282 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
#line 283 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
#line 284 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
#line 285 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_WARP_SIZE,
#line 286 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_PITCH,
#line 287 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
#line 288 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
#line 289 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
#line 290 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
#line 291 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
#line 292 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
#line 293 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
#line 294 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_INTEGRATED,
#line 295 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
#line 296 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
#line 297 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
#line 298 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
#line 299 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
#line 300 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
#line 301 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
#line 302 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
#line 303 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
#line 304 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
#line 305 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
#line 306 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
#line 307 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT,
#line 308 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES,
#line 309 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
#line 310 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
#line 311 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
#line 312 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
#line 313 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
#line 314 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
#line 315 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
#line 316 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
#line 317 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
#line 318 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
#line 319 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
#line 320 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
#line 321 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
#line 322 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
#line 323 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER,
#line 324 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
#line 325 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
#line 326 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
#line 327 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
#line 328 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
#line 329 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
#line 330 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
#line 331 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH,
#line 332 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
#line 333 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS,
#line 334 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
#line 335 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
#line 336 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
#line 337 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
#line 338 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
#line 339 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
#line 340 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,
#line 341 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS,
#line 342 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,
#line 343 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
#line 344 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS,
#line 345 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH,
#line 346 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
#line 347 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
#line 348 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
#line 349 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
#line 350 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
#line 351 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH,
#line 352 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH,
#line 353 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT,
#line 354 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
#line 355 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
#line 356 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH,
#line 357 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
#line 358 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_DEVICE_ATTRIBUTE_MAX};
#line 380 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUpointer_attribute_enum {
#line 381 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_POINTER_ATTRIBUTE_CONTEXT = 1,
#line 382 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
#line 383 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
#line 384 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_POINTER_ATTRIBUTE_HOST_POINTER,
#line 385 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_POINTER_ATTRIBUTE_P2P_TOKENS};
#line 391 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUfunction_attribute_enum {
#line 397 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
#line 404 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
#line 410 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
#line 415 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
#line 420 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_ATTRIBUTE_NUM_REGS,
#line 429 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_ATTRIBUTE_PTX_VERSION,
#line 438 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_ATTRIBUTE_BINARY_VERSION,
#line 440 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_ATTRIBUTE_MAX};
#line 446 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUfunc_cache_enum {
#line 447 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_CACHE_PREFER_NONE,
#line 448 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_CACHE_PREFER_SHARED,
#line 449 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_CACHE_PREFER_L1,
#line 450 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_FUNC_CACHE_PREFER_EQUAL};
#line 456 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUsharedconfig_enum {
#line 457 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE,
#line 458 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE,
#line 459 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE};
#line 465 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUmemorytype_enum {
#line 466 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_MEMORYTYPE_HOST = 1,
#line 467 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_MEMORYTYPE_DEVICE,
#line 468 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_MEMORYTYPE_ARRAY,
#line 469 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_MEMORYTYPE_UNIFIED};
#line 475 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUcomputemode_enum {
#line 476 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_COMPUTEMODE_DEFAULT,
#line 477 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_COMPUTEMODE_EXCLUSIVE,
#line 478 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_COMPUTEMODE_PROHIBITED,
#line 479 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_COMPUTEMODE_EXCLUSIVE_PROCESS};
#line 485 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUjit_option_enum {
#line 492 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_MAX_REGISTERS,
#line 507 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_THREADS_PER_BLOCK,
#line 515 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_WALL_TIME,
#line 524 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_INFO_LOG_BUFFER,
#line 533 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
#line 542 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_ERROR_LOG_BUFFER,
#line 551 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
#line 559 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_OPTIMIZATION_LEVEL,
#line 567 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_TARGET_FROM_CUCONTEXT,
#line 575 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_TARGET,
#line 583 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_FALLBACK_STRATEGY,
#line 591 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_GENERATE_DEBUG_INFO,
#line 598 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_LOG_VERBOSE,
#line 605 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_GENERATE_LINE_INFO,
#line 613 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_CACHE_MODE,
#line 615 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_NUM_OPTIONS};
#line 622 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUjit_target_enum {
#line 624 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TARGET_COMPUTE_10,
#line 625 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TARGET_COMPUTE_11,
#line 626 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TARGET_COMPUTE_12,
#line 627 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TARGET_COMPUTE_13,
#line 628 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TARGET_COMPUTE_20,
#line 629 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TARGET_COMPUTE_21,
#line 630 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TARGET_COMPUTE_30,
#line 631 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_TARGET_COMPUTE_35,
#line 632 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_MAX_JIT_TARGET};
#line 638 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUjit_fallback_enum {
#line 640 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_PREFER_PTX,
#line 642 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_PREFER_BINARY};
#line 649 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUjit_cacheMode_enum {
#line 651 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_CACHE_OPTION_NONE,
#line 652 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_CACHE_OPTION_CG,
#line 653 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_CACHE_OPTION_CA};
#line 659 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUjitInputType_enum {
#line 665 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_INPUT_CUBIN,
#line 671 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_INPUT_PTX,
#line 677 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_INPUT_FATBINARY,
#line 683 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_INPUT_OBJECT,
#line 689 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_INPUT_LIBRARY,
#line 691 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_JIT_NUM_INPUT_TYPES};
#line 701 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUgraphicsRegisterFlags_enum {
#line 702 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_NONE,
#line 703 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
#line 704 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
#line 705 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4,
#line 706 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8};
#line 712 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUgraphicsMapResourceFlags_enum {
#line 713 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE,
#line 714 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY,
#line 715 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD};
#line 721 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUarray_cubemap_face_enum {
#line 722 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CUBEMAP_FACE_POSITIVE_X,
#line 723 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_X,
#line 724 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CUBEMAP_FACE_POSITIVE_Y,
#line 725 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_Y,
#line 726 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CUBEMAP_FACE_POSITIVE_Z,
#line 727 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_Z};
#line 733 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUlimit_enum {
#line 734 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_LIMIT_STACK_SIZE,
#line 735 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_LIMIT_PRINTF_FIFO_SIZE,
#line 736 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_LIMIT_MALLOC_HEAP_SIZE,
#line 737 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH,
#line 738 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT,
#line 739 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_LIMIT_MAX};
#line 745 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUresourcetype_enum {
#line 746 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RESOURCE_TYPE_ARRAY,
#line 747 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RESOURCE_TYPE_MIPMAPPED_ARRAY,
#line 748 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RESOURCE_TYPE_LINEAR,
#line 749 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RESOURCE_TYPE_PITCH2D};
#line 755 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum cudaError_enum {
#line 761 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_SUCCESS,
#line 767 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_INVALID_VALUE,
#line 773 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_OUT_OF_MEMORY,
#line 779 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_NOT_INITIALIZED,
#line 784 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_DEINITIALIZED,
#line 791 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_PROFILER_DISABLED,
#line 799 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_PROFILER_NOT_INITIALIZED,
#line 806 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_PROFILER_ALREADY_STARTED,
#line 813 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_PROFILER_ALREADY_STOPPED,
#line 819 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_NO_DEVICE = 100,
#line 825 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_INVALID_DEVICE,
#line 832 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_INVALID_IMAGE = 200,
#line 842 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_INVALID_CONTEXT,
#line 851 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
#line 856 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_MAP_FAILED = 205,
#line 861 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_UNMAP_FAILED,
#line 867 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_ARRAY_IS_MAPPED,
#line 872 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_ALREADY_MAPPED,
#line 880 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_NO_BINARY_FOR_GPU,
#line 885 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_ALREADY_ACQUIRED,
#line 890 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_NOT_MAPPED,
#line 896 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
#line 902 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_NOT_MAPPED_AS_POINTER,
#line 908 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_ECC_UNCORRECTABLE,
#line 914 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_UNSUPPORTED_LIMIT,
#line 921 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
#line 927 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
#line 932 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_INVALID_SOURCE = 300,
#line 937 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_FILE_NOT_FOUND,
#line 942 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
#line 947 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
#line 952 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_OPERATING_SYSTEM,
#line 959 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_INVALID_HANDLE = 400,
#line 966 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_NOT_FOUND = 500,
#line 975 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_NOT_READY = 600,
#line 986 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_LAUNCH_FAILED = 700,
#line 997 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
#line 1008 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_LAUNCH_TIMEOUT,
#line 1014 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
#line 1021 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
#line 1028 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
#line 1034 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
#line 1041 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_CONTEXT_IS_DESTROYED,
#line 1049 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_ASSERT,
#line 1056 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_TOO_MANY_PEERS,
#line 1062 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
#line 1068 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
#line 1073 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_NOT_PERMITTED = 800,
#line 1079 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_NOT_SUPPORTED,
#line 1084 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CUDA_ERROR_UNKNOWN = 999};
#line 1313 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
enum CUresourceViewFormat_enum {
#line 1315 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_NONE,
#line 1316 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_1X8,
#line 1317 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_2X8,
#line 1318 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_4X8,
#line 1319 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_1X8,
#line 1320 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_2X8,
#line 1321 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_4X8,
#line 1322 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_1X16,
#line 1323 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_2X16,
#line 1324 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_4X16,
#line 1325 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_1X16,
#line 1326 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_2X16,
#line 1327 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_4X16,
#line 1328 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_1X32,
#line 1329 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_2X32,
#line 1330 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UINT_4X32,
#line 1331 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_1X32,
#line 1332 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_2X32,
#line 1333 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SINT_4X32,
#line 1334 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_1X16,
#line 1335 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_2X16,
#line 1336 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_4X16,
#line 1337 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_1X32,
#line 1338 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_2X32,
#line 1339 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_FLOAT_4X32,
#line 1340 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC1,
#line 1341 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC2,
#line 1342 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC3,
#line 1343 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC4,
#line 1344 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SIGNED_BC4,
#line 1345 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC5,
#line 1346 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SIGNED_BC5,
#line 1347 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC6H,
#line 1348 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_SIGNED_BC6H,
#line 1349 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cuda.h"
CU_RES_VIEW_FORMAT_UNSIGNED_BC7};
#line 60 "../Include\\cutil.h"
enum CUTBoolean {
#line 62 "../Include\\cutil.h"
CUTFalse,
#line 63 "../Include\\cutil.h"
CUTTrue};
#line 75 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
enum cufftResult_t {
#line 76 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_SUCCESS,
#line 77 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_INVALID_PLAN,
#line 78 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_ALLOC_FAILED,
#line 79 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_INVALID_TYPE,
#line 80 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_INVALID_VALUE,
#line 81 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_INTERNAL_ERROR,
#line 82 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_EXEC_FAILED,
#line 83 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_SETUP_FAILED,
#line 84 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_INVALID_SIZE,
#line 85 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_UNALIGNED_DATA,
#line 86 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_INCOMPLETE_PARAMETER_LIST,
#line 87 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_INVALID_DEVICE,
#line 88 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_PARSE_ERROR,
#line 89 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_NO_WORKSPACE};
#line 114 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
enum cufftType_t {
#line 115 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_R2C = 42,
#line 116 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_C2R = 44,
#line 117 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_C2C = 41,
#line 118 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_D2Z = 106,
#line 119 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_Z2D = 108,
#line 120 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_Z2Z = 105};
#line 145 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
enum cufftCompatibility_t {
#line 146 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_COMPATIBILITY_NATIVE,
#line 147 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_COMPATIBILITY_FFTW_PADDING,
#line 148 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC,
#line 149 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\cufft.h"
CUFFT_COMPATIBILITY_FFTW_ALL};
#line 82 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
enum curandStatus {
#line 83 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_SUCCESS,
#line 84 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_VERSION_MISMATCH = 100,
#line 85 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_NOT_INITIALIZED,
#line 86 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_ALLOCATION_FAILED,
#line 87 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_TYPE_ERROR,
#line 88 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_OUT_OF_RANGE,
#line 89 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_LENGTH_NOT_MULTIPLE,
#line 90 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
#line 91 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_LAUNCH_FAILURE = 201,
#line 92 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_PREEXISTING_FAILURE,
#line 93 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_INITIALIZATION_FAILED,
#line 94 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_ARCH_MISMATCH,
#line 95 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_STATUS_INTERNAL_ERROR = 999};
#line 108 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
enum curandRngType {
#line 109 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_RNG_TEST,
#line 110 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_RNG_PSEUDO_DEFAULT = 100,
#line 111 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_RNG_PSEUDO_XORWOW,
#line 112 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_RNG_PSEUDO_MRG32K3A = 121,
#line 113 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_RNG_PSEUDO_MTGP32 = 141,
#line 114 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
#line 115 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_RNG_QUASI_DEFAULT = 200,
#line 116 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_RNG_QUASI_SOBOL32,
#line 117 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
#line 118 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_RNG_QUASI_SOBOL64,
#line 119 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_RNG_QUASI_SCRAMBLED_SOBOL64};
#line 132 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
enum curandOrdering {
#line 133 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_ORDERING_PSEUDO_BEST = 100,
#line 134 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_ORDERING_PSEUDO_DEFAULT,
#line 135 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_ORDERING_PSEUDO_SEEDED,
#line 136 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_ORDERING_QUASI_DEFAULT = 201};
#line 149 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
enum curandDirectionVectorSet {
#line 150 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
#line 151 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6,
#line 152 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_DIRECTION_VECTORS_64_JOEKUO6,
#line 153 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6};
#line 215 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
enum curandMethod {
#line 216 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_CHOOSE_BEST,
#line 217 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_ITR,
#line 218 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_KNUTH,
#line 219 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_HITR,
#line 220 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_M1,
#line 221 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_M2,
#line 222 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_BINARY_SEARCH,
#line 223 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_DISCRETE_GAUSS,
#line 224 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_REJECTION,
#line 225 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_DEVICE_API,
#line 226 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_FAST_REJECTION,
#line 227 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_3RD,
#line 228 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_DEFINITION,
#line 229 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\curand.h"
CURAND_POISSON};
#line 4 "e:\\dropbox\\visualstudio\\chimera\\source\\chimera\\Particles.cuh"
struct EmitterData;
#line 25 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
enum _ZNSt4errc4errcE {
#line 26 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc28address_family_not_supportedE = 102,
#line 27 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc14address_in_useE = 100,
#line 28 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc21address_not_availableE,
#line 29 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc17already_connectedE = 113,
#line 30 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc22argument_list_too_longE = 7,
#line 31 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc22argument_out_of_domainE = 33,
#line 32 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc11bad_addressE = 14,
#line 33 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc19bad_file_descriptorE = 9,
#line 34 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc11bad_messageE = 104,
#line 35 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc11broken_pipeE = 32,
#line 36 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc18connection_abortedE = 106,
#line 37 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc30connection_already_in_progressE = 103,
#line 38 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc18connection_refusedE = 107,
#line 39 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc16connection_resetE,
#line 40 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc17cross_device_linkE = 18,
#line 41 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc28destination_address_requiredE = 109,
#line 42 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc23device_or_resource_busyE = 16,
#line 43 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc19directory_not_emptyE = 41,
#line 44 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc23executable_format_errorE = 8,
#line 45 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc11file_existsE = 17,
#line 46 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc14file_too_largeE = 27,
#line 47 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc17filename_too_longE = 38,
#line 48 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc22function_not_supportedE = 40,
#line 49 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc16host_unreachableE = 110,
#line 50 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc18identifier_removedE,
#line 51 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc21illegal_byte_sequenceE = 42,
#line 52 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc34inappropriate_io_control_operationE = 25,
#line 53 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc11interruptedE = 4,
#line 54 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc16invalid_argumentE = 22,
#line 55 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc12invalid_seekE = 29,
#line 56 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc8io_errorE = 5,
#line 57 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc14is_a_directoryE = 21,
#line 58 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc12message_sizeE = 115,
#line 59 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc12network_downE,
#line 60 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc13network_resetE,
#line 61 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc19network_unreachableE,
#line 62 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc15no_buffer_spaceE,
#line 63 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc16no_child_processE = 10,
#line 64 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc7no_linkE = 121,
#line 65 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc17no_lock_availableE = 39,
#line 66 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc20no_message_availableE = 120,
#line 67 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc10no_messageE = 122,
#line 68 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc18no_protocol_optionE,
#line 69 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc18no_space_on_deviceE = 28,
#line 70 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc19no_stream_resourcesE = 124,
#line 71 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc25no_such_device_or_addressE = 6,
#line 72 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc14no_such_deviceE = 19,
#line 73 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc25no_such_file_or_directoryE = 2,
#line 74 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc15no_such_processE,
#line 75 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc15not_a_directoryE = 20,
#line 76 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc12not_a_socketE = 128,
#line 77 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc12not_a_streamE = 125,
#line 78 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc13not_connectedE,
#line 79 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc17not_enough_memoryE = 12,
#line 80 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc13not_supportedE = 129,
#line 81 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc18operation_canceledE = 105,
#line 82 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc21operation_in_progressE = 112,
#line 83 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc23operation_not_permittedE = 1,
#line 84 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc23operation_not_supportedE = 130,
#line 85 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc21operation_would_blockE = 140,
#line 86 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc10owner_deadE = 133,
#line 87 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc17permission_deniedE = 13,
#line 88 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc14protocol_errorE = 134,
#line 89 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc22protocol_not_supportedE,
#line 90 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc21read_only_file_systemE = 30,
#line 91 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc29resource_deadlock_would_occurE = 36,
#line 92 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc30resource_unavailable_try_againE = 11,
#line 93 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc19result_out_of_rangeE = 34,
#line 94 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc21state_not_recoverableE = 127,
#line 95 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc14stream_timeoutE = 137,
#line 96 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc14text_file_busyE = 139,
#line 97 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc9timed_outE = 138,
#line 98 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc29too_many_files_open_in_systemE = 23,
#line 99 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc19too_many_files_openE,
#line 100 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc14too_many_linksE = 31,
#line 101 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc29too_many_symbolic_link_levelsE = 114,
#line 102 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc15value_too_largeE = 132,
#line 103 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt4errc19wrong_protocol_typeE = 136};
#line 112 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
enum _ZNSt7io_errc7io_errcE {
#line 113 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
_ZNSt7io_errc6streamE = 1};
#line 543 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\yvals.h"
enum _ZSt14_Uninitialized {
#line 545 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\yvals.h"
_ZSt7_Noinit};
#line 31 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\limits"
enum _ZSt18float_denorm_style {
#line 32 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\limits"
_ZSt20denorm_indeterminate = (-1),
#line 33 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\limits"
_ZSt13denorm_absent,
#line 34 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\limits"
_ZSt14denorm_present};
#line 39 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\limits"
enum _ZSt17float_round_style {
#line 40 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\limits"
_ZSt19round_indeterminate = (-1),
#line 41 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\limits"
_ZSt17round_toward_zero,
#line 42 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\limits"
_ZSt16round_to_nearest,
#line 43 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\limits"
_ZSt21round_toward_infinity,
#line 44 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\limits"
_ZSt25round_toward_neg_infinity};
#line 18 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xatomic0.h"
enum _ZSt12memory_order {
#line 19 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xatomic0.h"
_ZSt20memory_order_relaxed,
#line 20 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xatomic0.h"
_ZSt20memory_order_consume,
#line 21 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xatomic0.h"
_ZSt20memory_order_acquire,
#line 22 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xatomic0.h"
_ZSt20memory_order_release,
#line 23 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xatomic0.h"
_ZSt20memory_order_acq_rel,
#line 24 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xatomic0.h"
_ZSt20memory_order_seq_cst};
#line 503 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xstring"
enum _ZNSt11_String_valISt13_Simple_typesIcEEUt_E {
#line 504 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xstring"
_ZNSt11_String_valISt13_Simple_typesIcEE9_BUF_SIZEE = 16};
#line 507 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xstring"
enum _ZNSt11_String_valISt13_Simple_typesIcEEUt0_E {
#line 508 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xstring"
_ZNSt11_String_valISt13_Simple_typesIcEE11_ALLOC_MASKE = 15};
#line 69 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
struct _ZNSt6locale2idE;
#line 797 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
enum _ZNSt12codecvt_baseUt_E {
#line 798 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt12codecvt_base2okE,
#line 798 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt12codecvt_base7partialE,
#line 798 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt12codecvt_base5errorE,
#line 798 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt12codecvt_base6noconvE};
#line 2075 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
enum _ZNSt10ctype_baseUt_E {
#line 2076 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5alnumE = 263,
#line 2076 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5alphaE = 259,
#line 2077 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5cntrlE = 32,
#line 2077 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5digitE = 4,
#line 2077 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5graphE = 279,
#line 2078 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5lowerE = 2,
#line 2078 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5printE = 471,
#line 2079 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5punctE = 16,
#line 2079 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5spaceE = 72,
#line 2079 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5upperE = 1,
#line 2080 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base6xdigitE = 128,
#line 2080 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
_ZNSt10ctype_base5blankE = 72};
#line 159 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
struct _ZSt14error_category;
#line 576 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
struct _ZSt23_Generic_error_category;
#line 597 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
struct _ZSt24_Iostream_error_category;
#line 620 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
struct _ZSt22_System_error_category;
#line 53 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiE11_Dummy_enumE {
#line 53 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE15_Dummy_enum_valE = 1};
#line 54 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiE9_FmtflagsE {
#line 56 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE8_FmtmaskE = 65535,
#line 56 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE8_FmtzeroE = 0};
#line 85 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiE8_IostateE {
#line 87 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE9_StatmaskE = 23};
#line 95 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiE9_OpenmodeE {
#line 97 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE9_OpenmaskE = 255};
#line 108 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiE8_SeekdirE {
#line 110 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE9_SeekmaskE = 3};
#line 117 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
enum _ZNSt5_IosbIiEUt_E {
#line 118 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
_ZNSt5_IosbIiE9_OpenprotE = 64};
#line 213 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
enum _ZNSt8ios_base5eventE {
#line 215 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
_ZNSt8ios_base11erase_eventE,
#line 215 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
_ZNSt8ios_base11imbue_eventE,
#line 215 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xiosbase"
_ZNSt8ios_base13copyfmt_eventE};
#line 503 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xstring"
enum _ZNSt11_String_valISt13_Simple_typesIwEEUt_E {
#line 504 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xstring"
_ZNSt11_String_valISt13_Simple_typesIwEE9_BUF_SIZEE = 8};
#line 507 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xstring"
enum _ZNSt11_String_valISt13_Simple_typesIwEEUt0_E {
#line 508 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xstring"
_ZNSt11_String_valISt13_Simple_typesIwEE11_ALLOC_MASKE = 7};
#line 428 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\crtdefs.h"
typedef unsigned long long size_t;
#line 1 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"





































#line 1 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\host_defines.h"





















































































#line 87 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\host_defines.h"










#line 98 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\host_defines.h"








































#line 139 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\host_defines.h"










#line 150 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\host_defines.h"






#line 157 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\host_defines.h"




#line 162 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\host_defines.h"










#line 174 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\host_defines.h"

















#line 192 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\host_defines.h"








#line 201 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\host_defines.h"


#line 204 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\host_defines.h"
#line 39 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"







#line 47 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"
typedef __declspec(__device_builtin_texture_type__) const void *__texture_type__;
typedef __declspec(__device_builtin_surface_type__) const void *__surface_type__;
#line 50 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"






























































#line 113 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"





























extern __declspec(__device__) void* malloc(size_t);
extern __declspec(__device__) void free(void*);

extern __declspec(__device__) void __assertfail(
  const void  *message,
  const void  *file,
  unsigned int line,
  const void  *function,
  size_t       charsize);















#line 167 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"













#line 181 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"













#line 195 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"
static __declspec(__device__) void _wassert(
  const unsigned short *_Message,
  const unsigned short *_File,
  unsigned              _Line)
{
  __assertfail(
    (const void *)_Message,
    (const void *)_File,
                  _Line,
    (const void *)0,
    sizeof(unsigned short));
}
#line 208 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"

#line 210 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"

#line 1 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\builtin_types.h"























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\device_types.h"




















































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\host_defines.h"










































































































































































































#line 204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\host_defines.h"
#line 54 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\device_types.h"







enum __declspec(__device_builtin__) cudaRoundMode
{
    cudaRoundNearest,
    cudaRoundZero,
    cudaRoundPosInf,
    cudaRoundMinInf
};

#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\device_types.h"
#line 57 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"




















































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\host_defines.h"










































































































































































































#line 204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\host_defines.h"
#line 54 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"



























































#line 114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"










enum __declspec(__device_builtin__) cudaError
{
    




    cudaSuccess                           =      0,
  
    



    cudaErrorMissingConfiguration         =      1,
  
    



    cudaErrorMemoryAllocation             =      2,
  
    



    cudaErrorInitializationError          =      3,
  
    







    cudaErrorLaunchFailure                =      4,
  
    






    cudaErrorPriorLaunchFailure           =      5,
  
    







    cudaErrorLaunchTimeout                =      6,
  
    






    cudaErrorLaunchOutOfResources         =      7,
  
    



    cudaErrorInvalidDeviceFunction        =      8,
  
    






    cudaErrorInvalidConfiguration         =      9,
  
    



    cudaErrorInvalidDevice                =     10,
  
    



    cudaErrorInvalidValue                 =     11,
  
    



    cudaErrorInvalidPitchValue            =     12,
  
    



    cudaErrorInvalidSymbol                =     13,
  
    


    cudaErrorMapBufferObjectFailed        =     14,
  
    


    cudaErrorUnmapBufferObjectFailed      =     15,
  
    



    cudaErrorInvalidHostPointer           =     16,
  
    



    cudaErrorInvalidDevicePointer         =     17,
  
    



    cudaErrorInvalidTexture               =     18,
  
    



    cudaErrorInvalidTextureBinding        =     19,
  
    




    cudaErrorInvalidChannelDescriptor     =     20,
  
    



    cudaErrorInvalidMemcpyDirection       =     21,
  
    







    cudaErrorAddressOfConstant            =     22,
  
    






    cudaErrorTextureFetchFailed           =     23,
  
    






    cudaErrorTextureNotBound              =     24,
  
    






    cudaErrorSynchronizationError         =     25,
  
    



    cudaErrorInvalidFilterSetting         =     26,
  
    



    cudaErrorInvalidNormSetting           =     27,
  
    





    cudaErrorMixedDeviceExecution         =     28,
  
    




    cudaErrorCudartUnloading              =     29,
  
    


    cudaErrorUnknown                      =     30,

    





    cudaErrorNotYetImplemented            =     31,
  
    






    cudaErrorMemoryValueTooLarge          =     32,
  
    




    cudaErrorInvalidResourceHandle        =     33,
  
    





    cudaErrorNotReady                     =     34,
  
    




    cudaErrorInsufficientDriver           =     35,
  
    










    cudaErrorSetOnActiveProcess           =     36,
  
    



    cudaErrorInvalidSurface               =     37,
  
    



    cudaErrorNoDevice                     =     38,
  
    



    cudaErrorECCUncorrectable             =     39,
  
    


    cudaErrorSharedObjectSymbolNotFound   =     40,
  
    


    cudaErrorSharedObjectInitFailed       =     41,
  
    



    cudaErrorUnsupportedLimit             =     42,
  
    



    cudaErrorDuplicateVariableName        =     43,
  
    



    cudaErrorDuplicateTextureName         =     44,
  
    



    cudaErrorDuplicateSurfaceName         =     45,
  
    







    cudaErrorDevicesUnavailable           =     46,
  
    


    cudaErrorInvalidKernelImage           =     47,
  
    





    cudaErrorNoKernelImageForDevice       =     48,
  
    










    cudaErrorIncompatibleDriverContext    =     49,
      
    




    cudaErrorPeerAccessAlreadyEnabled     =     50,
    
    




    cudaErrorPeerAccessNotEnabled         =     51,
    
    



    cudaErrorDeviceAlreadyInUse           =     54,

    




    cudaErrorProfilerDisabled             =     55,

    





    cudaErrorProfilerNotInitialized       =     56,

    




    cudaErrorProfilerAlreadyStarted       =     57,

    




     cudaErrorProfilerAlreadyStopped       =    58,

    





    cudaErrorAssert                        =    59,
  
    




    cudaErrorTooManyPeers                 =     60,
  
    



    cudaErrorHostMemoryAlreadyRegistered  =     61,
        
    



    cudaErrorHostMemoryNotRegistered      =     62,

    


    cudaErrorOperatingSystem              =     63,

    



    cudaErrorPeerAccessUnsupported        =     64,

    




    cudaErrorLaunchMaxDepthExceeded       =     65,

    





    cudaErrorLaunchFileScopedTex          =     66,

    





    cudaErrorLaunchFileScopedSurf         =     67,

    












    cudaErrorSyncDepthExceeded            =     68,

    









    cudaErrorLaunchPendingCountExceeded   =     69,
    
    


    cudaErrorNotPermitted                 =     70,

    



    cudaErrorNotSupported                 =     71,

    


    cudaErrorStartupFailure               =   0x7f,

    





    cudaErrorApiFailureBase               =  10000
};




enum __declspec(__device_builtin__) cudaChannelFormatKind
{
    cudaChannelFormatKindSigned           =   0,      
    cudaChannelFormatKindUnsigned         =   1,      
    cudaChannelFormatKindFloat            =   2,      
    cudaChannelFormatKindNone             =   3       
};




struct __declspec(__device_builtin__) cudaChannelFormatDesc
{
    int                        x; 
    int                        y; 
    int                        z; 
    int                        w; 
    enum cudaChannelFormatKind f; 
};




typedef struct cudaArray *cudaArray_t;




typedef const struct cudaArray *cudaArray_const_t;

struct cudaArray;




typedef struct cudaMipmappedArray *cudaMipmappedArray_t;




typedef const struct cudaMipmappedArray *cudaMipmappedArray_const_t;

struct cudaMipmappedArray;




enum __declspec(__device_builtin__) cudaMemoryType
{
    cudaMemoryTypeHost   = 1, 
    cudaMemoryTypeDevice = 2  
};




enum __declspec(__device_builtin__) cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      
    cudaMemcpyHostToDevice        =   1,      
    cudaMemcpyDeviceToHost        =   2,      
    cudaMemcpyDeviceToDevice      =   3,      
    cudaMemcpyDefault             =   4       
};





struct __declspec(__device_builtin__) cudaPitchedPtr
{
    void   *ptr;      
    size_t  pitch;    
    size_t  xsize;    
    size_t  ysize;    
};





struct __declspec(__device_builtin__) cudaExtent
{
    size_t width;     
    size_t height;    
    size_t depth;     
};





struct __declspec(__device_builtin__) cudaPos
{
    size_t x;     
    size_t y;     
    size_t z;     
};




struct __declspec(__device_builtin__) cudaMemcpy3DParms
{
    cudaArray_t            srcArray;  
    struct cudaPos         srcPos;    
    struct cudaPitchedPtr  srcPtr;    
  
    cudaArray_t            dstArray;  
    struct cudaPos         dstPos;    
    struct cudaPitchedPtr  dstPtr;    
  
    struct cudaExtent      extent;    
    enum cudaMemcpyKind    kind;      
};




struct __declspec(__device_builtin__) cudaMemcpy3DPeerParms
{
    cudaArray_t            srcArray;  
    struct cudaPos         srcPos;    
    struct cudaPitchedPtr  srcPtr;    
    int                    srcDevice; 
  
    cudaArray_t            dstArray;  
    struct cudaPos         dstPos;    
    struct cudaPitchedPtr  dstPtr;    
    int                    dstDevice; 
  
    struct cudaExtent      extent;    
};




struct cudaGraphicsResource;




enum __declspec(__device_builtin__) cudaGraphicsRegisterFlags
{
    cudaGraphicsRegisterFlagsNone             = 0,  
    cudaGraphicsRegisterFlagsReadOnly         = 1,   
    cudaGraphicsRegisterFlagsWriteDiscard     = 2,  
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,  
    cudaGraphicsRegisterFlagsTextureGather    = 8   
};




enum __declspec(__device_builtin__) cudaGraphicsMapFlags
{
    cudaGraphicsMapFlagsNone         = 0,  
    cudaGraphicsMapFlagsReadOnly     = 1,  
    cudaGraphicsMapFlagsWriteDiscard = 2   
};




enum __declspec(__device_builtin__) cudaGraphicsCubeFace 
{
    cudaGraphicsCubeFacePositiveX = 0x00, 
    cudaGraphicsCubeFaceNegativeX = 0x01, 
    cudaGraphicsCubeFacePositiveY = 0x02, 
    cudaGraphicsCubeFaceNegativeY = 0x03, 
    cudaGraphicsCubeFacePositiveZ = 0x04, 
    cudaGraphicsCubeFaceNegativeZ = 0x05  
};




enum __declspec(__device_builtin__) cudaResourceType
{
    cudaResourceTypeArray          = 0x00, 
    cudaResourceTypeMipmappedArray = 0x01, 
    cudaResourceTypeLinear         = 0x02, 
    cudaResourceTypePitch2D        = 0x03  
};




enum __declspec(__device_builtin__) cudaResourceViewFormat
{
    cudaResViewFormatNone                      = 0x00, 
    cudaResViewFormatUnsignedChar1             = 0x01, 
    cudaResViewFormatUnsignedChar2             = 0x02, 
    cudaResViewFormatUnsignedChar4             = 0x03, 
    cudaResViewFormatSignedChar1               = 0x04, 
    cudaResViewFormatSignedChar2               = 0x05, 
    cudaResViewFormatSignedChar4               = 0x06, 
    cudaResViewFormatUnsignedShort1            = 0x07, 
    cudaResViewFormatUnsignedShort2            = 0x08, 
    cudaResViewFormatUnsignedShort4            = 0x09, 
    cudaResViewFormatSignedShort1              = 0x0a, 
    cudaResViewFormatSignedShort2              = 0x0b, 
    cudaResViewFormatSignedShort4              = 0x0c, 
    cudaResViewFormatUnsignedInt1              = 0x0d, 
    cudaResViewFormatUnsignedInt2              = 0x0e, 
    cudaResViewFormatUnsignedInt4              = 0x0f, 
    cudaResViewFormatSignedInt1                = 0x10, 
    cudaResViewFormatSignedInt2                = 0x11, 
    cudaResViewFormatSignedInt4                = 0x12, 
    cudaResViewFormatHalf1                     = 0x13, 
    cudaResViewFormatHalf2                     = 0x14, 
    cudaResViewFormatHalf4                     = 0x15, 
    cudaResViewFormatFloat1                    = 0x16, 
    cudaResViewFormatFloat2                    = 0x17, 
    cudaResViewFormatFloat4                    = 0x18, 
    cudaResViewFormatUnsignedBlockCompressed1  = 0x19, 
    cudaResViewFormatUnsignedBlockCompressed2  = 0x1a, 
    cudaResViewFormatUnsignedBlockCompressed3  = 0x1b, 
    cudaResViewFormatUnsignedBlockCompressed4  = 0x1c, 
    cudaResViewFormatSignedBlockCompressed4    = 0x1d, 
    cudaResViewFormatUnsignedBlockCompressed5  = 0x1e, 
    cudaResViewFormatSignedBlockCompressed5    = 0x1f, 
    cudaResViewFormatUnsignedBlockCompressed6H = 0x20, 
    cudaResViewFormatSignedBlockCompressed6H   = 0x21, 
    cudaResViewFormatUnsignedBlockCompressed7  = 0x22  
};




struct __declspec(__device_builtin__) cudaResourceDesc {
	enum cudaResourceType resType;             
	
	union {
		struct {
			cudaArray_t array;                 
		} array;
        struct {
            cudaMipmappedArray_t mipmap;       
        } mipmap;
		struct {
			void *devPtr;                      
			struct cudaChannelFormatDesc desc; 
			size_t sizeInBytes;                
		} linear;
		struct {
			void *devPtr;                      
			struct cudaChannelFormatDesc desc; 
			size_t width;                      
			size_t height;                     
			size_t pitchInBytes;               
		} pitch2D;
	} res;
};




struct __declspec(__device_builtin__) cudaResourceViewDesc
{
    enum cudaResourceViewFormat format;           
    size_t                      width;            
    size_t                      height;           
    size_t                      depth;            
    unsigned int                firstMipmapLevel; 
    unsigned int                lastMipmapLevel;  
    unsigned int                firstLayer;       
    unsigned int                lastLayer;        
};




struct __declspec(__device_builtin__) cudaPointerAttributes
{
    



    enum cudaMemoryType memoryType;

    








    int device;

    



    void *devicePointer;

    



    void *hostPointer;
};




struct __declspec(__device_builtin__) cudaFuncAttributes
{
   




   size_t sharedSizeBytes;

   



   size_t constSizeBytes;

   


   size_t localSizeBytes;

   




   int maxThreadsPerBlock;

   


   int numRegs;

   




   int ptxVersion;

   




   int binaryVersion;
};




enum __declspec(__device_builtin__) cudaFuncCache
{
    cudaFuncCachePreferNone   = 0,    
    cudaFuncCachePreferShared = 1,    
    cudaFuncCachePreferL1     = 2,    
    cudaFuncCachePreferEqual  = 3     
};





enum __declspec(__device_builtin__) cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
};




enum __declspec(__device_builtin__) cudaComputeMode
{
    cudaComputeModeDefault          = 0,  
    cudaComputeModeExclusive        = 1,  
    cudaComputeModeProhibited       = 2,  
    cudaComputeModeExclusiveProcess = 3   
};




enum __declspec(__device_builtin__) cudaLimit
{
    cudaLimitStackSize                    = 0x00, 
    cudaLimitPrintfFifoSize               = 0x01, 
    cudaLimitMallocHeapSize               = 0x02, 
    cudaLimitDevRuntimeSyncDepth          = 0x03, 
    cudaLimitDevRuntimePendingLaunchCount = 0x04  
};




enum __declspec(__device_builtin__) cudaOutputMode
{
    cudaKeyValuePair    = 0x00, 
    cudaCSV             = 0x01  
};




enum __declspec(__device_builtin__) cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock             = 1,  
    cudaDevAttrMaxBlockDimX                   = 2,  
    cudaDevAttrMaxBlockDimY                   = 3,  
    cudaDevAttrMaxBlockDimZ                   = 4,  
    cudaDevAttrMaxGridDimX                    = 5,  
    cudaDevAttrMaxGridDimY                    = 6,  
    cudaDevAttrMaxGridDimZ                    = 7,  
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,  
    cudaDevAttrTotalConstantMemory            = 9,  
    cudaDevAttrWarpSize                       = 10, 
    cudaDevAttrMaxPitch                       = 11, 
    cudaDevAttrMaxRegistersPerBlock           = 12, 
    cudaDevAttrClockRate                      = 13, 
    cudaDevAttrTextureAlignment               = 14, 
    cudaDevAttrGpuOverlap                     = 15, 
    cudaDevAttrMultiProcessorCount            = 16, 
    cudaDevAttrKernelExecTimeout              = 17, 
    cudaDevAttrIntegrated                     = 18, 
    cudaDevAttrCanMapHostMemory               = 19, 
    cudaDevAttrComputeMode                    = 20, 
    cudaDevAttrMaxTexture1DWidth              = 21, 
    cudaDevAttrMaxTexture2DWidth              = 22, 
    cudaDevAttrMaxTexture2DHeight             = 23, 
    cudaDevAttrMaxTexture3DWidth              = 24, 
    cudaDevAttrMaxTexture3DHeight             = 25, 
    cudaDevAttrMaxTexture3DDepth              = 26, 
    cudaDevAttrMaxTexture2DLayeredWidth       = 27, 
    cudaDevAttrMaxTexture2DLayeredHeight      = 28, 
    cudaDevAttrMaxTexture2DLayeredLayers      = 29, 
    cudaDevAttrSurfaceAlignment               = 30, 
    cudaDevAttrConcurrentKernels              = 31, 
    cudaDevAttrEccEnabled                     = 32, 
    cudaDevAttrPciBusId                       = 33, 
    cudaDevAttrPciDeviceId                    = 34, 
    cudaDevAttrTccDriver                      = 35, 
    cudaDevAttrMemoryClockRate                = 36, 
    cudaDevAttrGlobalMemoryBusWidth           = 37, 
    cudaDevAttrL2CacheSize                    = 38, 
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39, 
    cudaDevAttrAsyncEngineCount               = 40, 
    cudaDevAttrUnifiedAddressing              = 41,     
    cudaDevAttrMaxTexture1DLayeredWidth       = 42, 
    cudaDevAttrMaxTexture1DLayeredLayers      = 43, 
    cudaDevAttrMaxTexture2DGatherWidth        = 45, 
    cudaDevAttrMaxTexture2DGatherHeight       = 46, 
    cudaDevAttrMaxTexture3DWidthAlt           = 47, 
    cudaDevAttrMaxTexture3DHeightAlt          = 48, 
    cudaDevAttrMaxTexture3DDepthAlt           = 49, 
    cudaDevAttrPciDomainId                    = 50, 
    cudaDevAttrTexturePitchAlignment          = 51, 
    cudaDevAttrMaxTextureCubemapWidth         = 52, 
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53, 
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54, 
    cudaDevAttrMaxSurface1DWidth              = 55, 
    cudaDevAttrMaxSurface2DWidth              = 56, 
    cudaDevAttrMaxSurface2DHeight             = 57, 
    cudaDevAttrMaxSurface3DWidth              = 58, 
    cudaDevAttrMaxSurface3DHeight             = 59, 
    cudaDevAttrMaxSurface3DDepth              = 60, 
    cudaDevAttrMaxSurface1DLayeredWidth       = 61, 
    cudaDevAttrMaxSurface1DLayeredLayers      = 62, 
    cudaDevAttrMaxSurface2DLayeredWidth       = 63, 
    cudaDevAttrMaxSurface2DLayeredHeight      = 64, 
    cudaDevAttrMaxSurface2DLayeredLayers      = 65, 
    cudaDevAttrMaxSurfaceCubemapWidth         = 66, 
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67, 
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68, 
    cudaDevAttrMaxTexture1DLinearWidth        = 69, 
    cudaDevAttrMaxTexture2DLinearWidth        = 70, 
    cudaDevAttrMaxTexture2DLinearHeight       = 71, 
    cudaDevAttrMaxTexture2DLinearPitch        = 72, 
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73, 
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74, 
    cudaDevAttrComputeCapabilityMajor         = 75,  
    cudaDevAttrComputeCapabilityMinor         = 76, 
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77, 
    cudaDevAttrStreamPrioritiesSupported      = 78  
};




struct __declspec(__device_builtin__) cudaDeviceProp
{
    char   name[256];                  
    size_t totalGlobalMem;             
    size_t sharedMemPerBlock;          
    int    regsPerBlock;               
    int    warpSize;                   
    size_t memPitch;                   
    int    maxThreadsPerBlock;         
    int    maxThreadsDim[3];           
    int    maxGridSize[3];             
    int    clockRate;                  
    size_t totalConstMem;              
    int    major;                      
    int    minor;                      
    size_t textureAlignment;           
    size_t texturePitchAlignment;      
    int    deviceOverlap;              
    int    multiProcessorCount;        
    int    kernelExecTimeoutEnabled;   
    int    integrated;                 
    int    canMapHostMemory;           
    int    computeMode;                
    int    maxTexture1D;               
    int    maxTexture1DMipmap;         
    int    maxTexture1DLinear;         
    int    maxTexture2D[2];            
    int    maxTexture2DMipmap[2];      
    int    maxTexture2DLinear[3];      
    int    maxTexture2DGather[2];      
    int    maxTexture3D[3];            
    int    maxTexture3DAlt[3];         
    int    maxTextureCubemap;          
    int    maxTexture1DLayered[2];     
    int    maxTexture2DLayered[3];     
    int    maxTextureCubemapLayered[2];
    int    maxSurface1D;               
    int    maxSurface2D[2];            
    int    maxSurface3D[3];            
    int    maxSurface1DLayered[2];     
    int    maxSurface2DLayered[3];     
    int    maxSurfaceCubemap;          
    int    maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;           
    int    concurrentKernels;          
    int    ECCEnabled;                 
    int    pciBusID;                   
    int    pciDeviceID;                
    int    pciDomainID;                
    int    tccDriver;                  
    int    asyncEngineCount;           
    int    unifiedAddressing;          
    int    memoryClockRate;            
    int    memoryBusWidth;             
    int    l2CacheSize;                
    int    maxThreadsPerMultiProcessor;
    int    streamPrioritiesSupported;  
};




































































typedef __declspec(__device_builtin__) struct __declspec(__device_builtin__) cudaIpcEventHandle_st
{
    char reserved[64];
}cudaIpcEventHandle_t;




typedef __declspec(__device_builtin__) struct __declspec(__device_builtin__) cudaIpcMemHandle_st 
{
    char reserved[64];
}cudaIpcMemHandle_t;










typedef __declspec(__device_builtin__) enum cudaError cudaError_t;




typedef __declspec(__device_builtin__) struct CUstream_st *cudaStream_t;




typedef __declspec(__device_builtin__) struct CUevent_st *cudaEvent_t;




typedef __declspec(__device_builtin__) struct cudaGraphicsResource *cudaGraphicsResource_t;




typedef __declspec(__device_builtin__) struct CUuuid_st cudaUUID_t;




typedef __declspec(__device_builtin__) enum cudaOutputMode cudaOutputMode_t;


 

#line 1324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"

#line 58 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\surface_types.h"


























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"










































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"

#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\surface_types.h"
























enum __declspec(__device_builtin__) cudaSurfaceBoundaryMode
{
    cudaBoundaryModeZero  = 0,    
    cudaBoundaryModeClamp = 1,    
    cudaBoundaryModeTrap  = 2     
};




enum __declspec(__device_builtin__)  cudaSurfaceFormatMode
{
    cudaFormatModeForced = 0,     
    cudaFormatModeAuto = 1        
};




struct __declspec(__device_builtin__) surfaceReference
{
    


    struct cudaChannelFormatDesc channelDesc;
};




typedef __declspec(__device_builtin__) unsigned long long cudaSurfaceObject_t;


 

#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\surface_types.h"
#line 59 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_types.h"


























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"










































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"

#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_types.h"
























enum __declspec(__device_builtin__) cudaTextureAddressMode
{
    cudaAddressModeWrap   = 0,    
    cudaAddressModeClamp  = 1,    
    cudaAddressModeMirror = 2,    
    cudaAddressModeBorder = 3     
};




enum __declspec(__device_builtin__) cudaTextureFilterMode
{
    cudaFilterModePoint  = 0,     
    cudaFilterModeLinear = 1      
};




enum __declspec(__device_builtin__) cudaTextureReadMode
{
    cudaReadModeElementType     = 0,  
    cudaReadModeNormalizedFloat = 1   
};




struct __declspec(__device_builtin__) textureReference
{
    


    int                          normalized;
    


    enum cudaTextureFilterMode   filterMode;
    


    enum cudaTextureAddressMode  addressMode[3];
    


    struct cudaChannelFormatDesc channelDesc;
    


    int                          sRGB;
    


    unsigned int                 maxAnisotropy;
    


    enum cudaTextureFilterMode   mipmapFilterMode;
    


    float                        mipmapLevelBias;
    


    float                        minMipmapLevelClamp;
    


    float                        maxMipmapLevelClamp;
    int                          __cudaReserved[15];
};




struct __declspec(__device_builtin__) cudaTextureDesc
{
    


    enum cudaTextureAddressMode addressMode[3];
    


    enum cudaTextureFilterMode  filterMode;
    


    enum cudaTextureReadMode    readMode;
    


    int                         sRGB;
    


    int                         normalizedCoords;
    


    unsigned int                maxAnisotropy;
    


    enum cudaTextureFilterMode  mipmapFilterMode;
    


    float                       mipmapLevelBias;
    


    float                       minMipmapLevelClamp;
    


    float                       maxMipmapLevelClamp;
};




typedef __declspec(__device_builtin__) unsigned long long cudaTextureObject_t;


 

#line 214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_types.h"
#line 60 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"



























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\builtin_types.h"























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\device_types.h"




































































#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\device_types.h"
#line 57 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"










































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\surface_types.h"






















































































































#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\surface_types.h"
#line 59 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_types.h"




















































































































































































































#line 214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_types.h"
#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"












































































































































































































































































































































































































































#line 430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
#line 61 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\builtin_types.h"
#line 61 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
#line 62 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\host_defines.h"










































































































































































































#line 204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\host_defines.h"
#line 63 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"






















#line 87 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"







#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"

struct __declspec(__device_builtin__) char1
{
    signed char x;
};

struct __declspec(__device_builtin__) uchar1
{
    unsigned char x;
};


struct __declspec(__device_builtin__) __declspec(align(2)) char2
{
    signed char x, y;
};

struct __declspec(__device_builtin__) __declspec(align(2)) uchar2
{
    unsigned char x, y;
};

struct __declspec(__device_builtin__) char3
{
    signed char x, y, z;
};

struct __declspec(__device_builtin__) uchar3
{
    unsigned char x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(4)) char4
{
    signed char x, y, z, w;
};

struct __declspec(__device_builtin__) __declspec(align(4)) uchar4
{
    unsigned char x, y, z, w;
};

struct __declspec(__device_builtin__) short1
{
    short x;
};

struct __declspec(__device_builtin__) ushort1
{
    unsigned short x;
};

struct __declspec(__device_builtin__) __declspec(align(4)) short2
{
    short x, y;
};

struct __declspec(__device_builtin__) __declspec(align(4)) ushort2
{
    unsigned short x, y;
};

struct __declspec(__device_builtin__) short3
{
    short x, y, z;
};

struct __declspec(__device_builtin__) ushort3
{
    unsigned short x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(8)) short4 { short x; short y; short z; short w; };
struct __declspec(__device_builtin__) __declspec(align(8)) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; };

struct __declspec(__device_builtin__) int1
{
    int x;
};

struct __declspec(__device_builtin__) uint1
{
    unsigned int x;
};

struct __declspec(__device_builtin__) __declspec(align(8)) int2 { int x; int y; };
struct __declspec(__device_builtin__) __declspec(align(8)) uint2 { unsigned int x; unsigned int y; };

struct __declspec(__device_builtin__) int3
{
    int x, y, z;
};

struct __declspec(__device_builtin__) uint3
{
    unsigned int x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(16)) int4
{
    int x, y, z, w;
};

struct __declspec(__device_builtin__) __declspec(align(16)) uint4
{
    unsigned int x, y, z, w;
};

struct __declspec(__device_builtin__) long1
{
    long int x;
};

struct __declspec(__device_builtin__) ulong1
{
    unsigned long x;
};


struct __declspec(__device_builtin__) __declspec(align(8)) long2 { long int x; long int y; };
struct __declspec(__device_builtin__) __declspec(align(8)) ulong2 { unsigned long int x; unsigned long int y; };












#line 229 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"

struct __declspec(__device_builtin__) long3
{
    long int x, y, z;
};

struct __declspec(__device_builtin__) ulong3
{
    unsigned long int x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(16)) long4
{
    long int x, y, z, w;
};

struct __declspec(__device_builtin__) __declspec(align(16)) ulong4
{
    unsigned long int x, y, z, w;
};

struct __declspec(__device_builtin__) float1
{
    float x;
};















#line 271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"

struct __declspec(__device_builtin__) __declspec(align(8)) float2 { float x; float y; };

#line 275 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"


struct __declspec(__device_builtin__) float3
{
    float x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(16)) float4
{
    float x, y, z, w;
};

struct __declspec(__device_builtin__) longlong1
{
    long long int x;
};

struct __declspec(__device_builtin__) ulonglong1
{
    unsigned long long int x;
};

struct __declspec(__device_builtin__) __declspec(align(16)) longlong2
{
    long long int x, y;
};

struct __declspec(__device_builtin__) __declspec(align(16)) ulonglong2
{
    unsigned long long int x, y;
};

struct __declspec(__device_builtin__) longlong3
{
    long long int x, y, z;
};

struct __declspec(__device_builtin__) ulonglong3
{
    unsigned long long int x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(16)) longlong4
{
    long long int x, y, z ,w;
};

struct __declspec(__device_builtin__) __declspec(align(16)) ulonglong4
{
    unsigned long long int x, y, z, w;
};

struct __declspec(__device_builtin__) double1
{
    double x;
};

struct __declspec(__device_builtin__) __declspec(align(16)) double2
{
    double x, y;
};

struct __declspec(__device_builtin__) double3
{
    double x, y, z;
};

struct __declspec(__device_builtin__) __declspec(align(16)) double4
{
    double x, y, z, w;
};





#line 353 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"







typedef __declspec(__device_builtin__) struct char1 char1;
typedef __declspec(__device_builtin__) struct uchar1 uchar1;
typedef __declspec(__device_builtin__) struct char2 char2;
typedef __declspec(__device_builtin__) struct uchar2 uchar2;
typedef __declspec(__device_builtin__) struct char3 char3;
typedef __declspec(__device_builtin__) struct uchar3 uchar3;
typedef __declspec(__device_builtin__) struct char4 char4;
typedef __declspec(__device_builtin__) struct uchar4 uchar4;
typedef __declspec(__device_builtin__) struct short1 short1;
typedef __declspec(__device_builtin__) struct ushort1 ushort1;
typedef __declspec(__device_builtin__) struct short2 short2;
typedef __declspec(__device_builtin__) struct ushort2 ushort2;
typedef __declspec(__device_builtin__) struct short3 short3;
typedef __declspec(__device_builtin__) struct ushort3 ushort3;
typedef __declspec(__device_builtin__) struct short4 short4;
typedef __declspec(__device_builtin__) struct ushort4 ushort4;
typedef __declspec(__device_builtin__) struct int1 int1;
typedef __declspec(__device_builtin__) struct uint1 uint1;
typedef __declspec(__device_builtin__) struct int2 int2;
typedef __declspec(__device_builtin__) struct uint2 uint2;
typedef __declspec(__device_builtin__) struct int3 int3;
typedef __declspec(__device_builtin__) struct uint3 uint3;
typedef __declspec(__device_builtin__) struct int4 int4;
typedef __declspec(__device_builtin__) struct uint4 uint4;
typedef __declspec(__device_builtin__) struct long1 long1;
typedef __declspec(__device_builtin__) struct ulong1 ulong1;
typedef __declspec(__device_builtin__) struct long2 long2;
typedef __declspec(__device_builtin__) struct ulong2 ulong2;
typedef __declspec(__device_builtin__) struct long3 long3;
typedef __declspec(__device_builtin__) struct ulong3 ulong3;
typedef __declspec(__device_builtin__) struct long4 long4;
typedef __declspec(__device_builtin__) struct ulong4 ulong4;
typedef __declspec(__device_builtin__) struct float1 float1;
typedef __declspec(__device_builtin__) struct float2 float2;
typedef __declspec(__device_builtin__) struct float3 float3;
typedef __declspec(__device_builtin__) struct float4 float4;
typedef __declspec(__device_builtin__) struct longlong1 longlong1;
typedef __declspec(__device_builtin__) struct ulonglong1 ulonglong1;
typedef __declspec(__device_builtin__) struct longlong2 longlong2;
typedef __declspec(__device_builtin__) struct ulonglong2 ulonglong2;
typedef __declspec(__device_builtin__) struct longlong3 longlong3;
typedef __declspec(__device_builtin__) struct ulonglong3 ulonglong3;
typedef __declspec(__device_builtin__) struct longlong4 longlong4;
typedef __declspec(__device_builtin__) struct ulonglong4 ulonglong4;
typedef __declspec(__device_builtin__) struct double1 double1;
typedef __declspec(__device_builtin__) struct double2 double2;
typedef __declspec(__device_builtin__) struct double3 double3;
typedef __declspec(__device_builtin__) struct double4 double4;







struct __declspec(__device_builtin__) dim3
{
    unsigned int x, y, z;




#line 423 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
};

typedef __declspec(__device_builtin__) struct dim3 dim3;



#line 430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
#line 61 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\builtin_types.h"
#line 212 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"
#line 1 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\device_launch_parameters.h"




















































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"












































































































































































































































































































































































































































#line 430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
#line 54 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\device_launch_parameters.h"






#line 61 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\device_launch_parameters.h"



#line 65 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\device_launch_parameters.h"

uint3 __declspec(__device_builtin__) extern const threadIdx;
uint3 __declspec(__device_builtin__) extern const blockIdx;
dim3 __declspec(__device_builtin__) extern const blockDim;
dim3 __declspec(__device_builtin__) extern const gridDim;
int __declspec(__device_builtin__) extern const warpSize;





#line 77 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\device_launch_parameters.h"






#line 84 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\device_launch_parameters.h"






#line 91 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\device_launch_parameters.h"






#line 98 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\device_launch_parameters.h"






#line 105 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\device_launch_parameters.h"






#line 112 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\device_launch_parameters.h"

#line 114 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\device_launch_parameters.h"
#line 213 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"










































#line 44 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"






#line 51 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 55 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 59 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 63 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 67 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 71 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 75 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 79 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 83 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 87 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 91 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 99 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"



#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"

#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\crt\\storage_class.h"
#line 214 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\crt/device_runtime.h"
#line 430 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\crtdefs.h"
#line 677 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\driver_types.h"
typedef const struct cudaArray *cudaArray_const_t;
#line 74 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\cuda_texture_types.h"
struct _Z7textureI6float4Li3EL19cudaTextureReadMode0EE { struct textureReference __b_16textureReference;};
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
#pragma pack(8)
#line 69 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
struct _ZNSt6locale2idE {
#line 90 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\xlocale"
size_t _Id;};
#pragma pack()
#pragma pack(8)
#line 159 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
struct _ZSt14error_category { const long long *__vptr;};
#pragma pack()
#pragma pack(8)
#line 576 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
struct _ZSt23_Generic_error_category { struct _ZSt14error_category __b_St14error_category;};
#pragma pack()
#pragma pack(8)
#line 597 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
struct _ZSt24_Iostream_error_category { struct _ZSt23_Generic_error_category __b_St23_Generic_error_category;};
#pragma pack()
#pragma pack(8)
#line 620 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\system_error"
struct _ZSt22_System_error_category { struct _ZSt23_Generic_error_category __b_St23_Generic_error_category;};
#pragma pack()
#line 239 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
static  __declspec(__device__) __inline struct float3 _ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float3Efff(float, float, float);
#line 244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
static  __declspec(__device__) __inline struct float4 _ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float4Effff(float, float, float, float);

#line 247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 249 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 251 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 253 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 255 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 257 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 186 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double fabs(double);
#line 188 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 192 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 194 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 196 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 202 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 208 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 210 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 212 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 220 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 222 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 224 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 228 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 230 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 232 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 234 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 236 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 238 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 240 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 248 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 250 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 252 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 254 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 256 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 258 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 260 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 262 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 264 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 266 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 268 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 270 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 1679 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double ldexp(double, int);
#line 1681 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 1683 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 1685 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 1687 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 1689 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 1691 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 1693 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 1695 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 1697 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2252 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double frexp(double, int *);
#line 2254 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2256 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2258 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2260 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2262 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2264 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2266 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2268 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2270 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2276 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2278 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2280 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2282 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2284 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2286 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2288 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2290 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2292 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2294 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2296 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2298 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2300 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2302 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2304 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2306 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2308 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2310 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2312 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2314 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2316 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2318 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2320 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2322 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2326 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2328 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2330 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2332 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2334 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2336 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2338 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2340 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2342 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2344 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2346 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2348 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2352 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2354 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2356 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2358 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2360 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2362 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2366 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2368 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2370 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2372 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2374 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2376 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2378 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2380 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2382 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2384 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2386 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2388 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2390 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2392 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2394 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2396 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2398 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2400 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2402 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2404 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2406 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2408 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2410 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2412 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2414 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2416 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2418 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2420 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2422 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2424 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2426 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2428 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2432 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2434 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2436 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2438 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2440 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2442 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2444 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2446 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2448 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2450 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2452 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2454 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2456 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2458 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2460 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2462 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2464 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2466 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2468 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2470 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 2472 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 161 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) double hypot(double, double);
#line 163 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 166 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float hypotf(float, float);
#line 168 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 387 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float frexpf(float, int *);
#line 389 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 391 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float fabsf(float);
#line 393 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 395 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) float ldexpf(float, int);
#line 397 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 399 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 401 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 403 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 405 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 407 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 409 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 411 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 413 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 415 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 417 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 419 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 421 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 423 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 425 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 427 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 429 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 431 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 433 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 435 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 437 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 439 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 441 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 443 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 445 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 447 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 449 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 451 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 453 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 455 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 457 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 459 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 461 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 463 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 465 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 467 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 469 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 471 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 473 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 475 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 477 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 479 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 481 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 483 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 485 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 487 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 489 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 491 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 493 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 495 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 497 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 499 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 501 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 503 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 505 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 507 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 509 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 511 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 513 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 515 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 517 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 519 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 521 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 523 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 525 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 527 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 529 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 531 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 533 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 535 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 537 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 539 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 541 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 543 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 545 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 547 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 549 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 551 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 553 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 555 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 557 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 559 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 561 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 563 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 565 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 567 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 569 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 571 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 573 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 575 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 577 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 579 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 581 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 583 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 585 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 587 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 589 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 591 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 593 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 595 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 597 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 599 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 601 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 603 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 605 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 607 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 609 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 611 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 613 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 615 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 617 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 619 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 621 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 623 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 625 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 627 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 629 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 631 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 633 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 635 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 637 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 639 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 641 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 643 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 645 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 647 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 649 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 651 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 653 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 655 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 657 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 659 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 661 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 663 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 665 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 667 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 669 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 671 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 673 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 675 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 677 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 679 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 681 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 683 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 685 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 687 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 689 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 691 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 693 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 695 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 697 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 699 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 701 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 703 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 705 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 707 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 709 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 711 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 713 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 715 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 717 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 719 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 721 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 723 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 725 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 727 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 729 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 731 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 733 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 735 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 737 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 739 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 741 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 743 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 745 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 747 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 749 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 751 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 753 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 755 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 757 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 759 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 761 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 763 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 765 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 767 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 769 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 771 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 773 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 775 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 777 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 779 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 781 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 783 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 785 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 787 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 789 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 791 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 793 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 795 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 797 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 799 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 801 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 803 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 805 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 807 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 809 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 811 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 813 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 815 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 817 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 819 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 821 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 823 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 825 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 827 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 829 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 831 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 833 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 835 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 837 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 839 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 841 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 843 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 845 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 847 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 849 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 851 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 853 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 855 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 857 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 859 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 861 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 863 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 865 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 867 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 869 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 871 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 873 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 875 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 877 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 879 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 881 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 883 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 885 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 887 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 889 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 891 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 893 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 895 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 897 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 899 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 901 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 903 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 905 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 907 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 909 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 911 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 913 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 915 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 917 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 919 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 921 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 923 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 925 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 927 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 929 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 931 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 933 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 935 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 937 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 939 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 941 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 943 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 945 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 947 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 949 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 951 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 953 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 955 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 957 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 959 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 961 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 963 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 965 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 967 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 969 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 971 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 973 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 975 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 977 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 979 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 981 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 983 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 985 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 987 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 989 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 991 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 993 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 995 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 997 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 999 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1001 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1003 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1005 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1007 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1009 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1011 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1013 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1015 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1017 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1019 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1021 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1023 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1025 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1027 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1029 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1031 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1033 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1035 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1037 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1039 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1041 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1043 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1045 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1047 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1049 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1051 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1053 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1055 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1057 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1059 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1061 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1063 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1065 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1067 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1069 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1071 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1073 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1075 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1077 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1079 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1081 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1083 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1085 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1087 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1089 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1091 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1093 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1095 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1097 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1099 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1101 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1103 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1105 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1107 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1109 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1111 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1113 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1115 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1117 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1119 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1121 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1123 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1125 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1127 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1129 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1131 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1133 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1135 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1137 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1139 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1141 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1143 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1145 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1147 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1149 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1151 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1153 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1155 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1157 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1159 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1161 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1163 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1165 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1167 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1169 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1171 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1173 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1175 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1177 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1179 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1181 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1183 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1185 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1187 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1189 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1191 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1193 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1195 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1197 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1199 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1201 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1203 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1205 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1207 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1209 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1211 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1213 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1215 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1217 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1219 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1221 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1223 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1225 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1227 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1229 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1231 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1233 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1235 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1237 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1239 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1241 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1243 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1245 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1247 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1249 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1251 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1253 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1255 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1257 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1259 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1261 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1263 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1265 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1267 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1269 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1271 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1273 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1275 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1277 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1279 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1281 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1283 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1285 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1287 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1289 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1291 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1293 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1295 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1297 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1299 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1301 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1303 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1305 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1307 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1309 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1311 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1313 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1315 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1317 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1319 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1321 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1323 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1325 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1327 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1329 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1331 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1333 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1335 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1337 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1339 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1341 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1343 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1345 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1347 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1349 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1351 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1353 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1355 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1357 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1359 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1361 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1363 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1365 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1367 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1369 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1371 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1373 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1375 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1377 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1379 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1381 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1383 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1385 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1387 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1389 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1391 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1393 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1395 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1397 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1399 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1401 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1403 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1405 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1407 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1409 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1411 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1413 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1415 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1417 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1419 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1421 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1423 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1425 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1427 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1429 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1431 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1433 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1435 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1437 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1439 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1441 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1443 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1445 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1447 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1449 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1451 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1453 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1455 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1457 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1459 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1461 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1463 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1465 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1467 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1469 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1471 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1473 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1475 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1477 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1479 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1481 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1483 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1485 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1487 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1489 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1491 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1493 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1495 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1497 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1499 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1501 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1503 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1505 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1507 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1509 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1511 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1513 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1515 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1517 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1519 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1521 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1523 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1525 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1527 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"

#line 1529 "C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\\math.h"
#line 2631 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
static  __declspec(__device__) __forceinline struct float4 _ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc655tex3DE7textureI6float4Li3EL19cudaTextureReadMode0EEfff( __texture_type__, float, float, float);

#line 79 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
 __declspec(__device_builtin__) extern  __declspec(__device__) struct float4 __ftexfetch(__texture_type__, struct float4, int);
#line 81 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 83 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 85 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 87 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 89 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 91 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 93 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 97 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 99 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 113 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 129 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 133 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 137 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 141 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 151 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 153 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 155 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 157 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 159 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 161 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 165 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 167 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 175 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 177 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 179 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 183 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 187 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 191 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 195 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 197 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 199 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 201 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 203 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 207 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 209 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 211 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 213 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 215 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 217 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 219 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 221 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 223 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 225 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 229 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 231 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 233 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 235 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 237 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 239 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 243 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 245 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 249 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 251 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 253 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 255 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 257 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 259 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 261 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 263 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 265 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 267 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 269 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 273 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 275 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 277 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 279 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 281 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 283 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 285 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 287 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 289 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 291 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 293 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 295 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 297 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 299 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 301 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 303 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 305 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 307 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 309 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 311 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 313 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 315 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 317 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 319 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 321 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 323 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 325 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 327 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 329 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 331 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
#line 343 "../Include\\cutil_math.h"
extern  __declspec(__device__) void _ZpLR6float3S_(struct float3 *, struct float3);
#line 738 "../Include\\cutil_math.h"
extern  __declspec(__device__) void _ZmLR6float3f(struct float3 *, float);
#line 801 "../Include\\cutil_math.h"
extern  __declspec(__device__) void _ZmLR6float4f(struct float4 *, float);
#line 1123 "../Include\\cutil_math.h"
extern  __declspec(__device__) float _Z3dot6float3S_(struct float3, struct float3);
#line 21 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
extern  __declspec(__device__) int _Z5IsOutP6float4P6float3S2_(struct float4 *, struct float3 *, struct float3 *);
#line 29 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
extern  __declspec(__device__) int _Z7IsAliveP6float4(struct float4 *);
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  extern void _computeEmitter(struct float4 *, struct float4 *, struct float3 *, struct float3 *, struct EmitterData *, struct float3, float, float, float, float, UINT);
#line 82 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  extern void _computeGravity(struct float4 *, struct float3 *, float);
#line 104 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  extern void _computeTurbulence(struct float4 *, struct float3 *, struct float3 *, UINT, UINT);
#line 130 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  extern void _integrate(struct float4 *, struct float3 *, struct float3 *, float);
#line 159 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  extern void _computeGradientField(struct float4 *, struct float3 *, struct float4);
#line 201 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  extern void _computeGravityField(struct float4 *, struct float3 *, struct float4, int, float);
#line 239 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  extern void _computeVelocityDamping(struct float4 *, struct float3 *, float);
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
extern  __declspec(__device__) void _ZN4dim3C1Ejjj(struct dim3 *const, unsigned, unsigned, unsigned);
extern  __declspec(__device__) void _ZN4dim3C2Ejjj(struct dim3 *const, unsigned, unsigned, unsigned);
#line 19 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
  __texture_type__ ct_gradientTexture;
#line 1 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\common_functions.h"



























































































































































#line 157 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\common_functions.h"








#line 166 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\common_functions.h"

#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"
















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 8242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 13682 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"





#line 13688 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"



#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions_dbl_ptx1.h"













































































































































































































































































































































































































































































































































#line 527 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions_dbl_ptx1.h"

#line 529 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions_dbl_ptx1.h"
#line 13692 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 13694 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"





#line 13700 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 13702 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\math_functions.h"

#line 168 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\common_functions.h"

#line 170 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/include\\common_functions.h"

#line 21 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
static  __declspec(__device__) const long long _ZTVSt14error_category[9] = {0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL};
static  __declspec(__device__) const long long _ZTVSt23_Generic_error_category[9] = {0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL};
static  __declspec(__device__) const long long _ZTVSt24_Iostream_error_category[9] = {0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL};
static  __declspec(__device__) const long long _ZTVSt22_System_error_category[9] = {0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL};
#line 239 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
static  __declspec(__device__) __inline struct float3 _ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float3Efff(
#line 239 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
float x, 
#line 239 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
float y, 
#line 239 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
float z){
#line 240 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
{
#line 241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
 struct float3 __cuda_local_var_41583_10_non_const_t;
#line 241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
(__cuda_local_var_41583_10_non_const_t.x) = x;
#line 241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
(__cuda_local_var_41583_10_non_const_t.y) = y;
#line 241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
(__cuda_local_var_41583_10_non_const_t.z) = z;
#line 241 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
return __cuda_local_var_41583_10_non_const_t;
#line 242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
}}
#line 244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
static  __declspec(__device__) __inline struct float4 _ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float4Effff(
#line 244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
float x, 
#line 244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
float y, 
#line 244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
float z, 
#line 244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
float w){
#line 245 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
{
#line 246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
 struct float4 __cuda_local_var_41588_10_non_const_t;
#line 246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
(__cuda_local_var_41588_10_non_const_t.x) = x;
#line 246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
(__cuda_local_var_41588_10_non_const_t.y) = y;
#line 246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
(__cuda_local_var_41588_10_non_const_t.z) = z;
#line 246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
(__cuda_local_var_41588_10_non_const_t.w) = w;
#line 246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
return __cuda_local_var_41588_10_non_const_t;
#line 247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
}}

#line 250 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 252 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 254 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 256 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 258 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 260 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 262 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 264 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 266 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 268 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 270 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 276 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 278 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 280 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 282 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 284 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 286 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 288 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 290 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 292 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 294 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 296 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 298 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 300 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 302 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 304 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 306 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 308 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 310 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 312 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 314 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 316 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 318 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 320 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 322 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 326 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 328 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 330 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 332 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 334 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 336 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 338 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 340 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 342 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 344 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 346 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 348 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 352 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 354 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 356 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 358 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 360 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 362 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 366 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 368 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 370 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 372 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 374 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 376 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 378 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 380 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 382 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 384 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 386 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 388 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 390 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 392 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 394 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 396 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 398 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 400 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 402 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 404 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 406 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 408 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 410 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 412 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 414 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 416 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 418 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 420 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 422 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 424 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 426 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 428 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 432 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 434 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 436 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 438 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 440 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 442 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 444 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 446 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 448 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 450 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 452 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 454 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 456 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 458 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 460 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 462 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 464 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 466 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 468 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 470 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 472 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 474 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 476 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 478 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 480 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 482 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 484 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 486 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 488 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 490 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 492 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 494 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 496 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 498 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 500 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 502 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 504 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 506 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 508 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 510 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 512 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 514 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 516 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 518 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 520 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 522 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 524 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 526 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 528 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 530 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 532 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 534 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 536 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 538 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 540 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 542 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 544 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 546 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 548 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 550 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 552 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 554 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 556 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 558 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 560 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 562 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 564 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 566 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 568 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 570 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 572 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 574 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 576 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 578 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 580 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 582 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 584 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 586 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 588 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 590 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 592 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 594 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 596 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 598 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 600 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 602 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 604 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 606 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 608 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 610 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 612 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 614 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 616 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 618 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 620 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 622 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 624 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 626 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 628 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 630 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 632 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 634 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 636 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 638 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 640 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 642 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 644 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 646 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 648 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 650 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 652 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 654 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 656 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 658 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 660 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 662 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 664 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 666 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 668 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 670 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 672 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 674 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 676 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 678 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 680 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 682 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 684 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 686 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 688 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 690 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 692 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 694 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 696 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 698 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 700 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 702 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 704 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 706 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 708 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 710 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 712 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 714 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 716 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 718 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 720 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 722 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 724 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 726 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 728 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 730 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 732 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 734 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 736 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 738 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 740 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 742 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 744 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 746 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 748 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 750 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 752 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 754 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 756 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 758 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 760 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 762 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 764 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 766 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 768 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 770 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 772 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 774 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 776 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 778 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 780 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 782 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 784 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 786 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 788 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 790 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 792 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 794 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 796 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 798 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 800 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 802 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 804 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 806 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 808 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 810 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 812 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 814 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 816 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 818 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 820 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 822 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 824 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 826 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 828 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 830 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 832 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 834 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 836 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 838 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 840 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 842 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 844 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 846 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 848 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 850 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 852 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 854 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 856 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 858 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 860 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 862 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 864 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 866 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 868 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 870 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 872 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 874 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 876 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 878 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 880 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 882 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 884 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 886 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 888 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 890 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 892 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 894 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 896 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 898 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 900 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 902 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 904 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 906 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 908 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 910 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 912 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 914 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 916 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 918 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 920 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 922 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 924 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 926 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 928 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 930 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 932 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 934 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 936 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 938 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 940 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 942 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 944 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 946 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 948 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 950 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 952 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 954 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 956 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 958 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 960 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 962 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 964 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 966 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 968 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 970 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 972 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 974 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 976 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 978 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 980 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 982 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 984 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 986 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 988 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 990 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 992 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 994 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 996 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 998 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1000 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1002 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1004 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1006 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1008 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1010 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1012 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1014 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1016 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1018 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1020 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1022 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1024 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1026 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1028 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1030 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1032 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1034 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1036 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1038 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1040 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1042 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1044 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1046 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1048 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1050 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1052 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1054 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1056 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1058 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1060 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1062 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1064 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1066 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1068 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1070 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1072 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1074 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1076 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1078 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1080 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1082 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1084 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1086 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1088 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1090 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1092 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1094 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1096 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1098 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1102 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1106 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1110 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1116 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1118 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1122 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1130 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1132 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1134 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1136 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1138 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1140 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1146 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1148 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1150 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1152 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1154 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1158 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1160 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1162 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1164 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1166 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1174 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1176 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1180 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1182 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1184 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1186 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1188 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1192 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1194 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1196 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1198 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1200 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1202 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1204 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1206 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1208 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1210 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1212 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1216 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1220 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1222 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1224 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1228 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1230 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1232 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1234 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1236 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1238 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1240 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1244 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1248 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1250 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1252 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1254 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1256 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1258 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1260 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1262 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1264 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1266 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1268 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1270 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1276 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1278 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1280 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1282 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1284 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1286 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1288 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1290 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1292 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1294 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1296 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1298 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1300 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1302 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1304 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1306 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1308 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1310 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1312 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1314 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1316 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1318 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1320 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1322 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1326 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1328 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1330 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1332 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1334 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1336 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1338 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1340 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1342 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1344 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1346 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1348 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1352 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1354 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1356 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1358 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1360 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1362 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1366 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1368 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1370 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1372 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1374 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1376 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1378 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1380 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1382 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1384 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1386 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1388 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1390 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1392 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1394 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1396 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1398 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1400 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1402 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1404 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1406 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1408 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1410 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1412 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1414 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1416 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1418 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1420 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1422 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1424 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1426 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1428 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1430 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1432 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1434 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1436 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1438 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1440 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1442 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1444 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1446 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1448 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1450 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1452 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1454 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1456 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1458 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1460 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1462 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1464 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1466 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1468 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1470 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1472 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1474 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1476 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1478 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1480 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1482 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1484 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1486 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1488 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1490 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1492 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1494 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1496 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1498 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1500 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1502 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1504 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1506 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1508 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1510 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1512 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1514 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1516 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1518 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1520 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1522 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1524 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1526 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1528 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1530 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1532 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1534 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1536 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1538 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1540 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1542 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1544 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1546 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1548 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1550 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1552 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1554 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1556 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1558 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1560 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1562 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1564 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1566 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1568 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1570 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1572 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1574 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1576 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1578 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1580 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1582 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1584 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1586 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1588 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1590 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1592 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1594 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1596 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1598 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1600 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1602 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1604 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1606 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1608 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1610 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1612 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1614 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1616 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1618 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1620 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1622 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1624 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1626 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1628 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1630 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1632 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1634 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1636 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1638 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1640 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1642 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1644 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1646 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1648 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1650 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1652 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1654 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1656 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1658 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1660 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1662 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1664 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1666 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1668 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1670 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1672 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1674 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1676 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1678 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1680 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1682 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1684 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1686 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1688 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1690 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1692 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1694 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1696 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1698 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1700 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1702 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1704 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1706 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1708 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1710 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1712 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1714 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1716 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1718 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1720 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1722 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1724 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1726 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"

#line 1728 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_functions.h"
#line 2631 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
static  __declspec(__device__) __forceinline struct float4 _ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc655tex3DE7textureI6float4Li3EL19cudaTextureReadMode0EEfff(
#line 2631 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
 __texture_type__ t, 
#line 2631 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
float x, 
#line 2631 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
float y, 
#line 2631 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
float z){
#line 2632 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
{
#line 2633 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
 struct float4 __cuda_local_var_117416_10_non_const_v;
#line 2633 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
__cuda_local_var_117416_10_non_const_v = (__ftexfetch(t, (_ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float4Effff(x, y, z, (0.0F))), 3));
#line 2635 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
return _ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float4Effff((__cuda_local_var_117416_10_non_const_v.x), (__cuda_local_var_117416_10_non_const_v.y), (__cuda_local_var_117416_10_non_const_v.z), (__cuda_local_var_117416_10_non_const_v.w));
#line 2636 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
}}

#line 2639 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2641 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2643 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2645 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2647 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2649 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2651 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2653 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2655 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2657 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2659 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2661 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2663 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2665 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2667 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2669 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2671 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2673 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2675 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2677 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2679 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2681 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2683 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2685 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2687 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2689 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2691 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2693 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2695 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2697 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2699 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2701 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2703 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2705 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2707 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2709 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2711 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2713 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2715 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2717 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2719 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2721 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2723 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2725 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2727 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2729 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2731 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2733 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2735 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2737 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2739 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2741 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2743 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2745 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2747 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2749 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2751 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2753 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2755 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2757 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2759 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2761 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2763 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2765 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2767 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2769 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2771 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2773 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2775 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2777 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2779 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2781 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2783 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2785 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2787 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2789 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2791 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2793 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2795 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2797 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2799 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2801 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2803 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2805 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2807 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2809 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2811 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2813 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2815 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2817 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2819 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2821 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2823 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2825 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2827 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2829 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2831 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2833 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2835 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2837 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2839 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2841 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2843 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2845 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2847 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2849 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2851 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2853 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2855 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2857 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2859 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2861 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2863 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2865 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2867 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2869 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2871 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2873 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2875 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2877 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2879 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2881 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2883 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2885 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2887 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"

#line 2889 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\texture_fetch_functions.h"
#line 343 "../Include\\cutil_math.h"
 __declspec(__device__) __inline void _ZpLR6float3S_(
#line 343 "../Include\\cutil_math.h"
struct float3 *a, 
#line 343 "../Include\\cutil_math.h"
struct float3 b){
#line 344 "../Include\\cutil_math.h"
{
#line 345 "../Include\\cutil_math.h"
(a->x) += (b.x);
#line 345 "../Include\\cutil_math.h"
(a->y) += (b.y);
#line 345 "../Include\\cutil_math.h"
(a->z) += (b.z); 
#line 346 "../Include\\cutil_math.h"
}}
#line 738 "../Include\\cutil_math.h"
 __declspec(__device__) __inline void _ZmLR6float3f(
#line 738 "../Include\\cutil_math.h"
struct float3 *a, 
#line 738 "../Include\\cutil_math.h"
float b){
#line 739 "../Include\\cutil_math.h"
{
#line 740 "../Include\\cutil_math.h"
(a->x) *= b;
#line 740 "../Include\\cutil_math.h"
(a->y) *= b;
#line 740 "../Include\\cutil_math.h"
(a->z) *= b; 
#line 741 "../Include\\cutil_math.h"
}}
#line 801 "../Include\\cutil_math.h"
 __declspec(__device__) __inline void _ZmLR6float4f(
#line 801 "../Include\\cutil_math.h"
struct float4 *a, 
#line 801 "../Include\\cutil_math.h"
float b){
#line 802 "../Include\\cutil_math.h"
{
#line 803 "../Include\\cutil_math.h"
(a->x) *= b;
#line 803 "../Include\\cutil_math.h"
(a->y) *= b;
#line 803 "../Include\\cutil_math.h"
(a->z) *= b;
#line 803 "../Include\\cutil_math.h"
(a->w) *= b; 
#line 804 "../Include\\cutil_math.h"
}}
#line 1123 "../Include\\cutil_math.h"
 __declspec(__device__) __inline float _Z3dot6float3S_(
#line 1123 "../Include\\cutil_math.h"
struct float3 a, 
#line 1123 "../Include\\cutil_math.h"
struct float3 b){
#line 1124 "../Include\\cutil_math.h"
{
#line 1125 "../Include\\cutil_math.h"
return (((a.x) * (b.x)) + ((a.y) * (b.y))) + ((a.z) * (b.z));
#line 1126 "../Include\\cutil_math.h"
}}
#line 21 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 __declspec(__device__) int _Z5IsOutP6float4P6float3S2_(
#line 21 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 *pos, 
#line 21 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *min, 
#line 21 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *max){
#line 22 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 23 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
return (int)(((((((min->x) > (pos->x)) || ((max->x) < (pos->x))) || ((min->y) > (pos->y))) || ((max->y) < (pos->y))) || ((min->z) > (pos->z))) || ((max->z) < (pos->z)));
#line 27 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}}
#line 29 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 __declspec(__device__) int _Z7IsAliveP6float4(
#line 29 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 *pos){
#line 30 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 31 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
return (int)(((double)(pos->w)) > (0.5));
#line 32 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}}
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  void _computeEmitter(
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 *positions, 
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 *startingPositions, 
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *velos, 
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *acc, 
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct EmitterData *ed, 
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 translation, 
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
float time, 
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
float dt, 
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
float start, 
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
float end, 
#line 34 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
UINT N){
#line 35 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 36 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 UINT __cuda_local_var_225268_10_non_const_id;
#line 37 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float4 __cuda_local_var_225269_12_non_const_pos;
#line 38 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct EmitterData __cuda_local_var_225270_17_non_const_data;
#line 36 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225268_10_non_const_id = (((blockDim.x) * (blockIdx.x)) + (threadIdx.x));
#line 37 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225269_12_non_const_pos = (positions[__cuda_local_var_225268_10_non_const_id]);
#line 38 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225270_17_non_const_data = (ed[__cuda_local_var_225268_10_non_const_id]);
#line 42 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
if (!(_Z7IsAliveP6float4((&__cuda_local_var_225269_12_non_const_pos))))
#line 43 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 44 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225270_17_non_const_data.time) += dt;
#line 45 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(ed[__cuda_local_var_225268_10_non_const_id]) = __cuda_local_var_225270_17_non_const_data;
#line 46 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 48 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
if ((time - (__cuda_local_var_225270_17_non_const_data.birthTime)) > end)
#line 49 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 50 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225269_12_non_const_pos = (startingPositions[__cuda_local_var_225268_10_non_const_id]);
#line 51 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(positions[__cuda_local_var_225268_10_non_const_id]) = __cuda_local_var_225269_12_non_const_pos;
#line 52 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225270_17_non_const_data.time) = (0.0F);
#line 53 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225270_17_non_const_data.birthTime) = time;
#line 54 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(ed[__cuda_local_var_225268_10_non_const_id]) = __cuda_local_var_225270_17_non_const_data;
#line 55 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(acc[__cuda_local_var_225268_10_non_const_id]) = (_ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float3Efff((0.0F), (0.0F), (0.0F)));
#line 56 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(velos[__cuda_local_var_225268_10_non_const_id]) = (_ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float3Efff((0.0F), (0.0F), (0.0F)));
#line 57 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
return;
#line 59 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
} else  {
#line 59 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
if (!(_Z7IsAliveP6float4((&__cuda_local_var_225269_12_non_const_pos))))
#line 60 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 61 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
if ((__cuda_local_var_225270_17_non_const_data.time) > (end * (__cuda_local_var_225270_17_non_const_data.rand)))
#line 62 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 63 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225269_12_non_const_pos.x) += (translation.x);
#line 64 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225269_12_non_const_pos.y) += (translation.y);
#line 65 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225269_12_non_const_pos.z) += (translation.z);
#line 66 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225269_12_non_const_pos.w) = (1.0F);
#line 67 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225270_17_non_const_data.birthTime) = time;
#line 68 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(ed[__cuda_local_var_225268_10_non_const_id]) = __cuda_local_var_225270_17_non_const_data;
#line 69 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(positions[__cuda_local_var_225268_10_non_const_id]) = __cuda_local_var_225269_12_non_const_pos;
#line 70 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 71 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
} } 
#line 72 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}}
#line 82 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  void _computeGravity(
#line 82 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 *positions, 
#line 82 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *acc, 
#line 82 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
float factor){
#line 83 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 84 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 UINT __cuda_local_var_225316_10_non_const_id;
#line 85 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float4 __cuda_local_var_225317_12_non_const_pos;
#line 84 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225316_10_non_const_id = (((blockDim.x) * (blockIdx.x)) + (threadIdx.x));
#line 85 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225317_12_non_const_pos = (positions[__cuda_local_var_225316_10_non_const_id]);
#line 87 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
if (!(_Z7IsAliveP6float4((&__cuda_local_var_225317_12_non_const_pos))))
#line 87 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 87 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
return;
#line 87 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 87 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
;
#line 90 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 91 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225323_16_non_const_a;
#line 91 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225323_16_non_const_a = (acc[__cuda_local_var_225316_10_non_const_id]);
#line 92 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225323_16_non_const_a.y) += factor;
#line 93 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(acc[__cuda_local_var_225316_10_non_const_id]) = __cuda_local_var_225323_16_non_const_a;
#line 94 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
} 
#line 95 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}}
#line 104 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  void _computeTurbulence(
#line 104 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 *positions, 
#line 104 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *acc, 
#line 104 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *dirs, 
#line 104 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
UINT randomCount, 
#line 104 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
UINT time){
#line 105 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 106 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 UINT __cuda_local_var_225338_10_non_const_id;
#line 107 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float4 __cuda_local_var_225339_12_non_const_pos;
#line 106 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225338_10_non_const_id = (((blockDim.x) * (blockIdx.x)) + (threadIdx.x));
#line 107 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225339_12_non_const_pos = (positions[__cuda_local_var_225338_10_non_const_id]);
#line 109 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
if (!(_Z7IsAliveP6float4((&__cuda_local_var_225339_12_non_const_pos))))
#line 109 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 109 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
return;
#line 109 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 109 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
;
#line 112 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 113 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225345_16_non_const_a;
#line 114 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 UINT __cuda_local_var_225346_14_non_const_dirsIndex;
#line 115 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225347_16_non_const_dir;
#line 113 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225345_16_non_const_a = (acc[__cuda_local_var_225338_10_non_const_id]);
#line 114 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225346_14_non_const_dirsIndex = ((__cuda_local_var_225338_10_non_const_id + time) % randomCount);
#line 115 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225347_16_non_const_dir = (dirs[__cuda_local_var_225346_14_non_const_dirsIndex]);
#line 116 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225345_16_non_const_a.x) += (__cuda_local_var_225347_16_non_const_dir.x);
#line 117 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225345_16_non_const_a.y) += (__cuda_local_var_225347_16_non_const_dir.y);
#line 118 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225345_16_non_const_a.z) += (__cuda_local_var_225347_16_non_const_dir.z);
#line 119 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(acc[__cuda_local_var_225338_10_non_const_id]) = __cuda_local_var_225345_16_non_const_a;
#line 120 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
} 
#line 121 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}}
#line 130 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  void _integrate(
#line 130 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 *positions, 
#line 130 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *acc, 
#line 130 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *velocity, 
#line 130 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
float dt){
#line 131 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 132 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 UINT __cuda_local_var_225364_10_non_const_id;
#line 134 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float4 __cuda_local_var_225366_12_non_const_pos;
#line 136 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225368_12_non_const_a;
#line 137 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225369_12_non_const_v;
#line 132 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225364_10_non_const_id = (((blockDim.x) * (blockIdx.x)) + (threadIdx.x));
#line 134 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225366_12_non_const_pos = (positions[__cuda_local_var_225364_10_non_const_id]);
#line 136 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225368_12_non_const_a = (acc[__cuda_local_var_225364_10_non_const_id]);
#line 137 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225369_12_non_const_v = (velocity[__cuda_local_var_225364_10_non_const_id]);
#line 140 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225369_12_non_const_v.x) += (((__cuda_local_var_225368_12_non_const_a.x) * dt) * dt);
#line 141 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225369_12_non_const_v.y) += (((__cuda_local_var_225368_12_non_const_a.y) * dt) * dt);
#line 142 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225369_12_non_const_v.z) += (((__cuda_local_var_225368_12_non_const_a.z) * dt) * dt);
#line 144 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225366_12_non_const_pos.x) += (((__cuda_local_var_225369_12_non_const_v.x) * dt) * dt);
#line 145 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225366_12_non_const_pos.y) += (((__cuda_local_var_225369_12_non_const_v.y) * dt) * dt);
#line 146 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225366_12_non_const_pos.z) += (((__cuda_local_var_225369_12_non_const_v.z) * dt) * dt);
#line 148 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(velocity[__cuda_local_var_225364_10_non_const_id]) = __cuda_local_var_225369_12_non_const_v;
#line 149 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(positions[__cuda_local_var_225364_10_non_const_id]) = __cuda_local_var_225366_12_non_const_pos; 
#line 150 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}}
#line 159 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  void _computeGradientField(
#line 159 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 *positions, 
#line 159 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *acc, 
#line 159 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 position){
#line 160 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 161 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 UINT __cuda_local_var_225393_10_non_const_id;
#line 162 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float4 __cuda_local_var_225394_12_non_const_pos;
#line 166 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225398_12_non_const_coord;
#line 167 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 float __cuda_local_var_225399_11_non_const_scale;
#line 168 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float4 __cuda_local_var_225400_12_non_const_grad;
#line 170 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225402_12_non_const_dist;
#line 172 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 float __cuda_local_var_225404_11_non_const_distanceSquared;
#line 174 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 float __cuda_local_var_225406_11_non_const_range;
#line 181 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 float __cuda_local_var_225413_11_non_const_t;
#line 182 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 float __cuda_local_var_225414_11_non_const_s;
#line 186 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225418_12_non_const_a;
#line 161 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225393_10_non_const_id = (((blockDim.x) * (blockIdx.x)) + (threadIdx.x));
#line 162 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225394_12_non_const_pos = (positions[__cuda_local_var_225393_10_non_const_id]);
#line 164 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
if (!(_Z7IsAliveP6float4((&__cuda_local_var_225394_12_non_const_pos))))
#line 164 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 164 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
return;
#line 164 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 164 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
;
#line 166 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225398_12_non_const_coord = (_ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float3Efff((__cuda_local_var_225394_12_non_const_pos.x), (__cuda_local_var_225394_12_non_const_pos.y), (__cuda_local_var_225394_12_non_const_pos.z)));
#line 167 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225399_11_non_const_scale = (position.w);
#line 168 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225400_12_non_const_grad = (_ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc655tex3DE7textureI6float4Li3EL19cudaTextureReadMode0EEfff(ct_gradientTexture, ((position.x) + ((__cuda_local_var_225398_12_non_const_coord.x) * __cuda_local_var_225399_11_non_const_scale)), ((position.y) + ((__cuda_local_var_225398_12_non_const_coord.y) * __cuda_local_var_225399_11_non_const_scale)), ((position.z) + ((__cuda_local_var_225398_12_non_const_coord.z) * __cuda_local_var_225399_11_non_const_scale))));
#line 170 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225402_12_non_const_dist = (_ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float3Efff(((__cuda_local_var_225394_12_non_const_pos.x) - (position.x)), ((position.y) - (position.y)), ((__cuda_local_var_225394_12_non_const_pos.z) - (position.z))));
#line 172 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225404_11_non_const_distanceSquared = (_Z3dot6float3S_(__cuda_local_var_225402_12_non_const_dist, __cuda_local_var_225402_12_non_const_dist));
#line 174 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225406_11_non_const_range = (30.0F);
#line 176 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
if (__cuda_local_var_225404_11_non_const_distanceSquared > (__cuda_local_var_225406_11_non_const_range * __cuda_local_var_225406_11_non_const_range))
#line 177 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 178 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
return;
#line 179 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 181 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225413_11_non_const_t = (5.0F);
#line 182 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225414_11_non_const_s = (__cuda_local_var_225413_11_non_const_t - (__cuda_local_var_225413_11_non_const_t * ( fdividef(__cuda_local_var_225404_11_non_const_distanceSquared , (__cuda_local_var_225406_11_non_const_range * __cuda_local_var_225406_11_non_const_range)))));
#line 184 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
_ZmLR6float4f((&__cuda_local_var_225400_12_non_const_grad), ((float)((0.5) * ((double)__cuda_local_var_225414_11_non_const_s))));
#line 186 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225418_12_non_const_a = (acc[__cuda_local_var_225393_10_non_const_id]);
#line 187 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225418_12_non_const_a.x) += (__cuda_local_var_225400_12_non_const_grad.x);
#line 188 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225418_12_non_const_a.y) += (__cuda_local_var_225400_12_non_const_grad.y);
#line 189 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(__cuda_local_var_225418_12_non_const_a.z) += (__cuda_local_var_225400_12_non_const_grad.z);
#line 190 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(acc[__cuda_local_var_225393_10_non_const_id]) = __cuda_local_var_225418_12_non_const_a; 
#line 191 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}}
#line 201 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  void _computeGravityField(
#line 201 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 *positions, 
#line 201 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *acc, 
#line 201 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 posNrange, 
#line 201 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
int repel, 
#line 201 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
float scale){
#line 202 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 203 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 UINT __cuda_local_var_225435_10_non_const_id;
#line 204 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float4 __cuda_local_var_225436_12_non_const_pos;
#line 208 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225440_12_non_const_gpos;
#line 210 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 float __cuda_local_var_225442_11_non_const_range;
#line 212 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225444_12_non_const_grad;
#line 214 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 float __cuda_local_var_225446_11_non_const_distanceSquared;
#line 221 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 float __cuda_local_var_225453_11_non_const_s;
#line 226 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225458_12_non_const_a;
#line 203 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225435_10_non_const_id = (((blockDim.x) * (blockIdx.x)) + (threadIdx.x));
#line 204 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225436_12_non_const_pos = (positions[__cuda_local_var_225435_10_non_const_id]);
#line 206 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
if (!(_Z7IsAliveP6float4((&__cuda_local_var_225436_12_non_const_pos))))
#line 206 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 206 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
return;
#line 206 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 206 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
;
#line 208 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225440_12_non_const_gpos = (_ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float3Efff((posNrange.x), (posNrange.y), (posNrange.z)));
#line 210 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225442_11_non_const_range = ((5.0F) * (posNrange.w));
#line 212 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225444_12_non_const_grad = (_ZN39_INTERNAL_17_Particles_cpp1_ii_610dbc6511make_float3Efff(((__cuda_local_var_225436_12_non_const_pos.x) - (__cuda_local_var_225440_12_non_const_gpos.x)), ((__cuda_local_var_225436_12_non_const_pos.y) - (__cuda_local_var_225440_12_non_const_gpos.y)), ((__cuda_local_var_225436_12_non_const_pos.z) - (__cuda_local_var_225440_12_non_const_gpos.z))));
#line 214 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225446_11_non_const_distanceSquared = (_Z3dot6float3S_(__cuda_local_var_225444_12_non_const_grad, __cuda_local_var_225444_12_non_const_grad));
#line 216 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
if (__cuda_local_var_225446_11_non_const_distanceSquared > (__cuda_local_var_225442_11_non_const_range * __cuda_local_var_225442_11_non_const_range))
#line 217 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 218 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
return;
#line 219 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 221 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225453_11_non_const_s = (((float)repel) - ( fdividef(__cuda_local_var_225446_11_non_const_distanceSquared , (__cuda_local_var_225442_11_non_const_range * __cuda_local_var_225442_11_non_const_range))));
#line 222 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225453_11_non_const_s *= scale;
#line 224 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
_ZmLR6float3f((&__cuda_local_var_225444_12_non_const_grad), __cuda_local_var_225453_11_non_const_s);
#line 226 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225458_12_non_const_a = (acc[__cuda_local_var_225435_10_non_const_id]);
#line 227 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
_ZpLR6float3S_((&__cuda_local_var_225458_12_non_const_a), __cuda_local_var_225444_12_non_const_grad);
#line 228 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(acc[__cuda_local_var_225435_10_non_const_id]) = __cuda_local_var_225458_12_non_const_a; 
#line 229 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}}
#line 239 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__declspec(__global__)  void _computeVelocityDamping(
#line 239 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float4 *positions, 
#line 239 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
struct float3 *velo, 
#line 239 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
float damping){
#line 240 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 241 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 UINT __cuda_local_var_225473_10_non_const_id;
#line 242 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float4 __cuda_local_var_225474_12_non_const_pos;
#line 246 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
 struct float3 __cuda_local_var_225478_12_non_const_a;
#line 241 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225473_10_non_const_id = (((blockDim.x) * (blockIdx.x)) + (threadIdx.x));
#line 242 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225474_12_non_const_pos = (positions[__cuda_local_var_225473_10_non_const_id]);
#line 244 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
if (!(_Z7IsAliveP6float4((&__cuda_local_var_225474_12_non_const_pos))))
#line 244 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
{
#line 244 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
return;
#line 244 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}
#line 244 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
;
#line 246 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
__cuda_local_var_225478_12_non_const_a = (velo[__cuda_local_var_225473_10_non_const_id]);
#line 247 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
_ZmLR6float3f((&__cuda_local_var_225478_12_non_const_a), damping);
#line 248 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
(velo[__cuda_local_var_225473_10_non_const_id]) = __cuda_local_var_225478_12_non_const_a; 
#line 249 "E:/Dropbox/VisualStudio/Chimera/Source/chimera/Particles.cu"
}}
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
 __declspec(__device__) __inline void _ZN4dim3C1Ejjj( struct dim3 *const this, 
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
unsigned vx, 
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
unsigned vy, 
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
unsigned vz){
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
{
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
(this->x) = vx;
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
(this->y) = vy;
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
(this->z) = vz; 
#line 419 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v5.5\\include\\vector_types.h"
}}
 __declspec(__device__) __inline void _ZN4dim3C2Ejjj( struct dim3 *const this,  unsigned __T20,  unsigned __T21,  unsigned __T22){ {  _ZN4dim3C1Ejjj(this, __T20, __T21, __T22);  }}

