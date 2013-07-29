#include "StdAfx.h"
#include "cudah.h"
#include <algorithm>
#include <cudaD3D11.h>
#include "GameApp.h"

#ifdef _DEBUG
#include "Process.h"
#include "GameApp.h"
#include "GameLogic.h"
#include "util.h"
#endif

#ifdef _DEBUG
    #pragma comment(lib, "cuda.lib")
#else
    #pragma comment(lib, "cuda.lib")
#endif

namespace cudah 
{
    //INT cudah::m_sDev = 0;
    //cudaDeviceProp cudah::m_sProps = cudaDeviceProp();
    CUcontext g_CudaContext;
    CUdevice g_device;
    cuda_stream g_defaultStream;

    BOOL Init(ID3D11Device* device) 
    {
        CUDA_DRIVER_SAFE_CALLING(cuInit(0));

        INT devices = 0;

        CUDA_DRIVER_SAFE_CALLING(cuDeviceGetCount(&devices));

	    //cudah::m_sDev = cutGetMaxGflopsDeviceId();
        CUDA_DRIVER_SAFE_CALLING(cuDeviceGet(&g_device, 0));

        /*cudaError_t error = cudaGetDeviceProperties(&cudah::m_sProps, cudah::m_sDev);
	    CUDA_RUNTIME_SAFE_CALLING(error);
        PrintDeviceInfo(cudah::m_sProps, TRUE); */
        /*
        printf("MaxThreads Dim per Block (%d, %d, %d)\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
        printf("MaxThreads per Block %d\n", props.maxThreadsPerBlock);
	    printf("Setting cuda cutGetMaxGflopsDevice: %d\n", dev);
	    printf("CC=%d.%d\n", props.major, props.minor); */
        //PrintDeviceInfo(props);

	    //CUDA_RUNTIME_SAFE_CALLING(cudaSetDevice(cudah::m_sDev));
        if(device)
        {
            CUDA_DRIVER_SAFE_CALLING(cuD3D11CtxCreate(&g_CudaContext, &g_device, CU_CTX_SCHED_BLOCKING_SYNC, device));
        }
        else
        {
            CUDA_DRIVER_SAFE_CALLING(cuCtxCreate(&g_CudaContext, CU_CTX_SCHED_BLOCKING_SYNC, g_device));
        }

        g_defaultStream = new _cuda_stream();
        g_defaultStream->Create();

        return TRUE;
    }

    VOID Destroy(VOID)
    {
        if(g_defaultStream)
        {
            g_defaultStream->Destroy();
        }
        SAFE_DELETE(g_defaultStream);
        if(g_CudaContext)
        {
            CUDA_DRIVER_SAFE_CALLING(cuCtxDestroy(g_CudaContext));
        }
       // cudaDeviceReset();
    }

    template<typename T>
    void copyLinearToPitchedMemory(T* devPtr, CONST T* linearData, UINT pitch, UINT width, UINT height, UINT depth)
    {
        UINT slicePitch = pitch * height;
        UINT index = 0;
        for (UINT z = 0; z < depth; ++z) 
        {
            T* slice = devPtr + z * slicePitch;

            for (UINT y = 0; y < height; ++y) 
            {
                T* row = slice + y * pitch;

                for (UINT x = 0; x < width; ++x) 
                {
                    row[x] = linearData[index++];
                }
            }
        }
    }

    cudah::cudah(LPCSTR file)
    { 
        std::string path = app::g_pApp->GetConfig()->GetString("sPTXPath");
        m_module.m_file = path + file;
        OnRestore();

#ifdef _DEBUG
        std::vector<std::string> p = util::split(std::string(file), '/');
        std::string& s = p[p.size()-1];
        std::string fn = util::split(s, '.')[0];
        fn += ".cu";
        std::wstring finalName(fn.begin(), fn.end());
        std::shared_ptr<proc::WatchCudaFileModificationProcess> modProc = 
            std::shared_ptr<proc::WatchCudaFileModificationProcess>(new proc::WatchCudaFileModificationProcess(this, finalName.c_str(), L"./chimera/"));
        app::g_pApp->GetLogic()->AttachProcess(modProc);
#endif

    }

    cudah::cudah(VOID)
    {
    }

    VOID cudah::OnRestore(VOID)
    {
        if(m_module.m_cudaModule)
        {
            cuModuleUnload(m_module.m_cudaModule);
        }

        CUDA_DRIVER_SAFE_CALLING(cuModuleLoad(&m_module.m_cudaModule, m_module.m_file.c_str()));

        TBD_FOR(m_module.m_kernel)
        {
            CUDA_DRIVER_SAFE_CALLING(cuModuleGetFunction(&it->second->m_fpCuda, m_module.m_cudaModule, it->second->m_func_name.c_str()));
        }

        std::vector<TextureBindung> bindung = m_textureBinding;
        m_textureBinding.clear();
        TBD_FOR(bindung)
        {
            BindArrayToTexture(it->m_textureName.c_str(), it->m_cudaArray, it->m_filterMode, it->m_addressMode, it->m_flags);
        }
    }

    cuda_kernel cudah::GetKernel(LPCSTR name)
    {
        auto it = m_module.m_kernel.find(name);

        if(it != m_module.m_kernel.end())
        {
            return it->second;
        }

        cuda_kernel kernel = new _cuda_kernel();

        kernel->m_func_name = std::string(name);

        kernel->m_stream = g_defaultStream;

        assert(m_module.m_cudaModule != NULL);

        CUDA_DRIVER_SAFE_CALLING(cuModuleGetFunction(&kernel->m_fpCuda, m_module.m_cudaModule, name));

        m_module.m_kernel[std::string(name)] = kernel;

        return kernel;
    }

    VOID cudah::BindArrayToTexture(LPCSTR textur, cuda_array array, CUfilter_mode filterMode, CUaddress_mode addressMode, UINT flags)
    {
        CUtexref ref;
        CUDA_DRIVER_SAFE_CALLING(cuModuleGetTexRef(&ref, m_module.m_cudaModule, textur));

        TextureBindung bind;
        bind.m_addressMode = addressMode;
        bind.m_cudaArray = array;
        bind.m_filterMode = filterMode;
        bind.m_flags = flags;
        bind.m_textureName = std::string(textur);
        m_textureBinding.push_back(bind);

        CUDA_DRIVER_SAFE_CALLING(cuTexRefSetArray(ref, array->m_array, CU_TRSA_OVERRIDE_FORMAT));
        CUDA_DRIVER_SAFE_CALLING(cuTexRefSetFlags(ref, flags));
        CUDA_DRIVER_SAFE_CALLING(cuTexRefSetFilterMode(ref, filterMode));
        CUDA_DRIVER_SAFE_CALLING(cuTexRefSetAddressMode(ref, 0, addressMode));
    }

    cuda_array cudah::CreateArray(CONST std::string& name, CONST UINT w, CONST UINT h, CONST UINT d, ArrayFormat format, FLOAT* data)
    {
        CUarray a;

        CUDA_ARRAY3D_DESCRIPTOR desc;
        desc.Format = CU_AD_FORMAT_FLOAT;
        desc.NumChannels = 1 + format;

        //cudaExtent ex = make_cudaExtent(w, h, d); 
        desc.Width = w;
        desc.Height = h;
        desc.Depth = d;
        desc.Flags = 0;
        
        CUDA_DRIVER_SAFE_CALLING(cuArray3DCreate(&a, &desc));

        //CUDA_RUNTIME_SAFE_CALLING(cudaMalloc3DArray(&array, &desc, ex)); //TODO: flag

        cuda_array c_array = new _cuda_array(a);
        //c_array->desc = desc;
        m_array[name] = c_array;

        if(data)
        {
            UINT hi = h == 0 ? 1 : h;
            UINT di = d == 0 ? 1 : h;
            UINT size = d * h * w;
            c_array->size = size;
            c_array->elements = size * desc.NumChannels * sizeof(FLOAT);
            
            /*UINT pitch = 512;
            FLOAT* pitched = new FLOAT[pitch * h * d];
            for(UINT i = 0; i < pitch * h * d; ++i)
            {
                pitched[i] = 0;
            }

            copyLinearToPitchedMemory<FLOAT>(pitched, data, pitch, w, h, d); */
    
            /*cudaMemcpy3DParms params = {0};
            ex = make_cudaExtent(w, h, d);
            params.dstArray = array;
            params.srcPtr = make_cudaPitchedPtr(data, ex.width * elementSize, ex.width, ex.height);
            params.extent = ex;
            params.kind = cudaMemcpyHostToDevice;
            */
            CUDA_MEMCPY3D cpy;
            ZeroMemory(&cpy, sizeof(CUDA_MEMCPY3D));
            cpy.Depth = d;
            cpy.WidthInBytes = w * desc.NumChannels * sizeof(FLOAT);
            cpy.Height = h;
            cpy.srcHeight = h;
            cpy.srcPitch = cpy.WidthInBytes;
            cpy.dstArray = a;
            cpy.srcHost = data;
            cpy.srcMemoryType = CU_MEMORYTYPE_HOST;
            cpy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            //cpy.
            CUDA_DRIVER_SAFE_CALLING(cuMemcpy3D(&cpy));
            //SAFE_ARRAY_DELETE(pitched);
        }

        return c_array;
    }

    cuda_buffer cudah::CreateBuffer(std::string& name, UINT byteSize, UINT elementSize) 
    {
    #if defined CUDA_SAFE
	    if(CheckResourceNotExists(name))
    #endif
	    {
		    CUdeviceptr ptr;
            CUDA_DRIVER_SAFE_CALLING(cuMemAlloc(&ptr, byteSize));

		    cuda_buffer b = new _cuda_buffer;
		    b->ptr = ptr;
		    b->size = byteSize;
            b->elements = b->size / elementSize;
		    b->isfromD3D = FALSE;
		    m_buffer.insert(std::pair<std::string, cuda_buffer>(name, b));
		    return b;
	    }
	    return NULL;
    }

    cuda_buffer cudah::CreateBuffer(std::string& name, UINT byteSize, VOID* data, UINT elementSize) 
    {
	    CreateBuffer(name, byteSize, elementSize);
	    cuda_buffer b = GetBuffer(name);
        WriteBuffer(b, data);
	    return b;
    }

    cuda_buffer cudah::RegisterD3D11Buffer(std::string& name, ID3D11Resource* resource, enum cudaGraphicsMapFlags flags)
    {

#if defined CUDA_SAFE
        CheckResourceNotExists(name);
#endif
        cuda_buffer b = new _cuda_buffer;
        b->isfromD3D = TRUE;
        
        CUDA_DRIVER_SAFE_CALLING(cuGraphicsD3D11RegisterResource(&b->glRes, resource, flags));

        m_buffer.insert(std::pair<std::string, cuda_buffer>(name, b));
        return b;
    }

    map_info cudah::MapGraphicsResource(cuda_buffer buffer) {

#if defined CUDA_SAFE
        if(!buffer->isfromD3D) {
            return map_info();
        }
#endif
        //CUDA_RUNTIME_SAFE_CALLING(cudaGraphicsMapResources(1, &buffer->glRes, 0));
        
        CUDA_DRIVER_SAFE_CALLING(cuGraphicsMapResources(1, &buffer->glRes, 0));
        map_info info;
        CUDA_DRIVER_SAFE_CALLING(cuGraphicsResourceGetMappedPointer(&(info.ptr), &(info.size), buffer->glRes));

        buffer->ptr = info.ptr;
        return info;
    }

    VOID cudah::UnmapGraphicsResource(cuda_buffer buffer) 
    {
        CUDA_DRIVER_SAFE_CALLING(cuGraphicsUnmapResources(1, &buffer->glRes, 0));
    }

    cuda_buffer cudah::GetBuffer(std::string& name) 
    {
    #if defined CUDA_SAFE
	    if(CheckResourceExists(name))
    #endif
	    {
		    return m_buffer.find(name)->second;
	    }
	    return NULL;
    }

    VOID cudah::ReadBuffer(std::string& name, VOID* dst) 
    {
    #if defined CUDA_SAFE
	    if(CheckResourceExists(name))
    #endif
	    {
		    cuda_buffer b = GetBuffer(name);
            CUDA_DRIVER_SAFE_CALLING(cuMemcpyDtoH(dst, b->ptr, b->size));
            //CUDA_DRIVER_SAFE_CALLING(cuMemcpy(dst, b->ptr, b->size, cudaMemcpyDeviceToHost));
	    }
    }

    VOID cudah::WriteBuffer(cuda_buffer buffer, VOID* src) 
    {
        CUDA_DRIVER_SAFE_CALLING(cuMemcpyHtoD(buffer->ptr, src, buffer->size));
    }

    VOID cudah::DeleteBuffer(std::string& name) 
    {
#if defined CUDA_SAFE
        if(CheckResourceExists(name))
#endif
        {
            auto it = m_buffer.find(name);   
            m_buffer.erase(name);
            it->second->VDestroy();
            delete it->second;
        }
    }

    cuda_record cudah::CreateRecord(VOID)
    {
        cuda_record record = new _cuda_record();
        m_records.push_back(record);
        return record;
    }

    cuda_record cudah::CreateRecord(cuda_stream stream)
    {
        cuda_record record = new _cuda_record(stream);
        m_records.push_back(record);
        return record;
    }

    VOID cudah::DestroyRecord(cuda_record record)
    {
        auto it = std::find(m_records.begin(), m_records.end(), record);
        m_records.erase(it);
        record->Destroy();
    }

    cuda_stream cudah::CreateStream(VOID)
    {
        cuda_stream stream = new _cuda_stream();
        stream->Create();
        m_streams.push_back(stream);
        return stream;
    }

    VOID cudah::Destroy() 
    {
        for(auto i = m_buffer.begin(); i != m_buffer.end(); ++i) 
        {
            cuda_buffer b = (*i).second;
            b->VDestroy();
            delete b;
        }
        m_buffer.clear();

        for(auto i = m_array.begin(); i != m_array.end(); ++i) 
        {
            cuda_array b = (*i).second;
            b->VDestroy();
            delete b;
        }
        m_array.clear();

        for(auto rc = m_records.begin(); rc != m_records.end(); ++rc)
        {
            (*rc)->Destroy();
            delete *rc;
        }
        m_records.clear();

        TBD_FOR(m_streams)
        {
            (*it)->Destroy();
            delete *it;
        }

        if(m_module.m_cudaModule)
        {
            cuModuleUnload(m_module.m_cudaModule);
        }

        TBD_FOR(m_module.m_kernel)
        {
            SAFE_DELETE(it->second);
        }
    }

    VOID cudah::CallKernel(cuda_kernel kernel)
    {
        CUDA_DRIVER_SAFE_CALLING(
            cuLaunchKernel(
            kernel->m_fpCuda, 
            kernel->m_gridDim.x, kernel->m_gridDim.y, kernel->m_gridDim.z, 
            kernel->m_blockDim.x, kernel->m_blockDim.y, kernel->m_blockDim.z,
            kernel->m_shrdMemBytes, kernel->m_stream->GetPtr(), kernel->m_ppArgs, kernel->m_ppExtras)
            );
    }

    cudah::~cudah(VOID)
    {
        this->Destroy();
    }
     /*
    VOID PrintDeviceInfo(cudaDeviceProp &prop, BOOL writeToFile) 
    {
       
        if(writeToFile) 
        {
            freopen ("cudaDeviceProperties.txt","w", stdout);
        }

        printf("Cuda Device Properties:\n");
        printf("ASCII string identifying device: %s \n", prop.name);
        printf("Global memory available on device in bytes: %d \n", prop.totalGlobalMem);
        printf("Shared memory available per block in bytes: %d \n", prop.sharedMemPerBlock);
        printf("32-bit registers available per block: %d \n", prop.regsPerBlock);
        printf("Warp size in threads: %d \n", prop.warpSize);
        printf("Maximum pitch in bytes allowed by memory copies: %d \n", prop.memPitch);
        printf("Maximum number of threads per block: %d \n", prop.maxThreadsPerBlock);
        printf("Maximum size of each dimension of a block: (%d, %d, %d) \n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid:  (%d, %d, %d) \n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Clock frequency in kilohertz: %d \n", prop.clockRate);
        printf("Constant memory available on device in bytes: %d \n", prop.totalConstMem);
        printf("Major compute capability: %d \n", prop.major);
        printf("Minor compute capability: %d \n", prop.minor);
        printf("Alignment requirement for textures: %d \n", prop.textureAlignment);
        printf("Pitch alignment requirement for texture references bound to pitched memory: %d \n", prop.texturePitchAlignment);
        printf("Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.: %d \n", prop.deviceOverlap);
        printf("Number of multiprocessors on device: %d \n", prop.multiProcessorCount);
        printf("Specified whether there is a run time limit on kernels: %d \n", prop.kernelExecTimeoutEnabled);
        printf("Device is integrated as opposed to discrete: %d \n", prop.integrated);
        printf("Device can map host memory with cudaHostAlloc or cudaHostGetDevicePointer: %d \n", prop.canMapHostMemory);
        printf("Compute mode (See ::cudaComputeMode): %d \n", prop.computeMode);
        printf("Maximum 1D texture size: %d \n", prop.maxTexture1D);
        printf("Maximum size for 1D textures bound to linear memory: %d \n", prop.maxTexture1DLinear);
        printf("Maximum 2D texture dimensions: %d \n", prop.maxTexture2D[2]);
        printf("Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory: (%d, %d, %d) \n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], prop.maxTexture2DLinear[2]);
        printf("Maximum 2D texture dimensions if texture gather operations have to be performed: (%d, %d) \n", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]);
        printf("Maximum 3D texture dimensions: (%d, %d, %d) \n", prop.maxTexture3D[0],  prop.maxTexture3D[1],  prop.maxTexture3D[2]);
        printf("Maximum Cubemap texture dimensions: %d \n", prop.maxTextureCubemap);
        printf("Maximum 1D layered texture dimensions: (%d, %d) \n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);
        printf("Maximum 2D layered texture dimensions: (%d, %d, %d) \n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
        printf("Maximum Cubemap layered texture dimensions: (%d, %d) \n", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);
        printf("Maximum 1D surface size: %d \n", prop.maxSurface1D);
        printf("Maximum 2D surface dimensions: (%d, %d) \n", prop.maxSurface2D[0], prop.maxSurface2D[1]);
        printf("Maximum 3D surface dimensions: (%d, %d, %d) \n", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);
        printf("Maximum 1D layered surface dimensions: (%d, %d, %d) \n", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]);
        printf("Maximum 2D layered surface dimensions: (%d, %d, %d) \n", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]);
        printf("Maximum Cubemap surface dimensions: %d \n", prop.maxSurfaceCubemap);
        printf("Maximum Cubemap layered surface dimensions: %d \n", prop.maxSurfaceCubemapLayered[2]);
        printf("Alignment requirements for surfaces: %d \n", prop.surfaceAlignment);
        printf("Device can possibly execute multiple kernels concurrently: %d \n", prop.concurrentKernels);
        printf("Device has ECC support enabled: %d \n", prop.ECCEnabled);
        printf("PCI bus ID of the device: %d \n", prop.pciBusID);
        printf("PCI device ID of the device: %d \n", prop.pciDeviceID);
        printf("PCI domain ID of the device: %d \n", prop.pciDomainID);
        printf("1 if device is a Tesla device using TCC driver, 0 otherwise: %d \n", prop.tccDriver);
        printf("Number of asynchronous engines: %d \n", prop.asyncEngineCount);
        printf("Device shares a unified address space with the host: %d \n", prop.unifiedAddressing);
        printf("Peak memory clock frequency in kilohertz: %d \n", prop.memoryClockRate);
        printf("Global memory bus width in bits: %d \n", prop.memoryBusWidth);
        printf("Size of L2 cache in bytes: %d \n", prop.l2CacheSize);
        printf("Maximum resident threads per multiprocessor: %d \n", prop.maxThreadsPerMultiProcessor);
        printf("Cuda Device Properties end\n");
        if(writeToFile)
        {
            fclose (stdout);
        }
    } */
}
