#pragma once
#include "stdafx.h"
#include <map>
#include <cuda.h>
#include <vector>
#include "d3d.h"
#include <vector_types.h>
#include <drvapi_error_string.h>

namespace cudah
{
#ifdef _DEBUG
#define CUDA_SAFE
#endif

#ifdef CUDA_SAFE 
#define CUDA_DRIVER_SAFE_CALLING(__error__) {  \
    if (CUDA_SUCCESS != __error__) { \
    LOG_CRITICAL_ERROR_A("%s\n", getCudaDrvErrorString( __error__)); \
    } }
#else
#define CUDA_DRIVER_SAFE_CALLING(__error__) __error__
#endif

    class _cuda_stream
    {
        friend class cudah;
        friend class _cuda_record;

    protected:
        CUstream m_stream;


    public:
        _cuda_stream(VOID) : m_stream(NULL)
        {

        }

        VOID Destroy(VOID)
        {
            if(m_stream)
            {
                cuStreamDestroy(m_stream);
            }
        }

        VOID Create(VOID)
        {
            cuStreamCreate(&m_stream, 0);
        }

        CUstream GetPtr(VOID)
        {
            return m_stream;
        }
    };

    typedef _cuda_stream* cuda_stream;

    class _cuda_record 
    {
        friend class cudah;
    private:
        INT m_start, m_stop;
        cuda_stream m_stream;
        FLOAT m_millis;
        _cuda_record(cuda_stream stream) : m_millis(-1.0f), m_start(0), m_stop(0), m_stream(stream)
        {
            Create();
        }

        _cuda_record(VOID) : m_millis(-1.0f), m_start(0), m_stop(0), m_stream(0)
        {
            Create();
        }

        VOID Create(VOID)
        {
            /*CUDA_RUNTIME_SAFE_CALLING(cudaEventCreate(&m_start));
            CUDA_RUNTIME_SAFE_CALLING(cudaEventCreate(&m_stop)); */
        }

        VOID StartRecording(VOID)
        {
            //CUDA_RUNTIME_SAFE_CALLING(cudaEventRecord(m_start, m_stream->m_stream));
        }

        VOID EndRecording(VOID)
        {
            /*CUDA_RUNTIME_SAFE_CALLING(cudaEventRecord(m_stop, m_stream->m_stream));
            CUDA_RUNTIME_SAFE_CALLING(cudaEventSynchronize(m_stop));
            CUDA_RUNTIME_SAFE_CALLING(cudaEventElapsedTime(&m_millis, m_start, m_stop)); */
        }

        FLOAT GetElapsedTimeMS(VOID)
        {
            return m_millis;
        }

        VOID Destroy(VOID)
        {
           /* if(m_start)
            {
                CUDA_RUNTIME_SAFE_CALLING(cudaEventDestroy(m_start));
            }

            if(m_stop)
            {
                CUDA_RUNTIME_SAFE_CALLING(cudaEventDestroy(m_stop));
            } */

            m_start = 0;
            m_stop = 0;
        }
    };

    typedef _cuda_record* cuda_record;

    enum ResType
    {
        eARRAY,
        eBuffer
    };
    class cuda_resource
    {
    protected:
        UINT size;
        UINT elements;

        virtual ResType VGetType(VOID) CONST = 0;
        virtual VOID VDestroy(VOID) = 0;
    public:
        virtual UINT VGetByteCount(VOID) CONST = 0;
        virtual UINT VGetElementCount(VOID) CONST = 0;
    };

    class _cuda_buffer : public cuda_resource
    {
        friend class cudah;

    protected:
        BOOL isfromD3D;
        INT flag;
        CUgraphicsResource glRes;

        _cuda_buffer(int i) {
            ptr = 0;
        }

        _cuda_buffer() {
            ptr = 0;
        }

        VOID VDestroy(VOID)
        {
            if(isfromD3D) 
            {
                CUDA_DRIVER_SAFE_CALLING(cuGraphicsUnregisterResource(glRes));
            } else {
                CUDA_DRIVER_SAFE_CALLING(cuMemFree(ptr));
            }
        }

        ResType VGetType(VOID) CONST
        {
            return eBuffer;
        }

    public:
        CUdeviceptr ptr;

        UINT VGetByteCount(VOID) CONST { return size; }
        UINT VGetElementCount(VOID) CONST { return elements; }
    };

    typedef _cuda_buffer* cuda_buffer;

    class _cuda_array : public cuda_resource
    {
        friend class cudah;
    protected:
        //cudaChannelFormatDesc desc;

        _cuda_array(CUarray array) : m_array(array)
        {

        }

        ResType VGetType(VOID) CONST
        {
            return eARRAY;
        }

        VOID VDestroy(VOID)
        {
            if(m_array)
            {
                CUDA_DRIVER_SAFE_CALLING(cuArrayDestroy(m_array));
            }
        }
    public:
        UINT VGetByteCount(VOID) CONST { return size; }
        UINT VGetElementCount(VOID) CONST { return elements; }
        //CONST cudaChannelFormatDesc& GetChannelDesc(VOID) CONST { return desc; }
        CUarray m_array;
    };

    typedef _cuda_array* cuda_array;

    struct map_info 
    {
        size_t size;
        CUdeviceptr ptr;
    };

    enum ArrayFormat
    {
        eR,
        eRG,
        eRGB,
        eRGBA
    };

    typedef class _cuda_kernel
    {
        friend class cudah;
    private:
        _cuda_kernel(VOID) : m_fpCuda(NULL), m_shrdMemBytes(0), m_ppExtras(NULL), m_ppArgs(NULL)
        {
            m_gridDim.x = 1;
            m_gridDim.y = 1;
            m_gridDim.z = 1;
            m_blockDim.x = 1;
            m_blockDim.y = 1;
            m_blockDim.z = 1;
        }

        CUfunction m_fpCuda;
        std::string m_func_name;

    public:
        dim3 m_gridDim;
        dim3 m_blockDim;
        UINT m_shrdMemBytes;
        cuda_stream m_stream;
        VOID** m_ppExtras;
        VOID** m_ppArgs;
    } *cuda_kernel;

    typedef class _cuda_module
    {
        friend class cudah;
    private:
        _cuda_module(VOID) : m_cudaModule(NULL) {}
        std::string m_file;
        std::map<std::string, cuda_kernel> m_kernel;
        CUmodule m_cudaModule;
    } *cuda_module;

    BOOL Init(ID3D11Device* device);

    VOID Destroy(VOID);

    //VOID PrintDeviceInfo(cudaDeviceProp& prop, BOOL writeToFile = 0);

    typedef struct _TextureBindung
    {
        std::string m_textureName;
        cuda_array m_cudaArray;
        CUfilter_mode m_filterMode;
        CUaddress_mode m_addressMode;
        UINT m_flags;
    } TextureBindung;

    //cuda stuff
    class cudah
    {
    private:
        std::map<std::string, cuda_buffer> m_buffer;
        std::map<std::string, cuda_array> m_array;
        std::vector<TextureBindung> m_textureBinding;
        std::list<cuda_record> m_records;
        std::list<cuda_stream> m_streams;
        std::string m_modFile;
        _cuda_module m_module;

    public:
        //static CUdevice m_sDev;
        //static cudaDeviceProp m_sProps;
        //static CUcontext m_pCudaContext;

        cudah(LPCSTR mod);

        cudah(VOID);

        VOID Destroy(VOID);

        VOID OnRestore(VOID);

        cuda_buffer CreateBuffer(std::string&, UINT byteSize, UINT elementByteSize);

        cuda_buffer CreateBuffer(std::string&, UINT byteSize, VOID* data, UINT elementByteSize);

        cuda_buffer RegisterD3D11Buffer(std::string& name, ID3D11Resource* res, enum cudaGraphicsMapFlags flags);

        cuda_array CreateArray(CONST std::string& name, CONST UINT w, CONST UINT h, CONST UINT d, ArrayFormat format, FLOAT* data = NULL);

        VOID DeleteBuffer(std::string& buffer);

        VOID DeleteBuffer(cuda_buffer buffer);

        VOID DeleteArray(std::string& buffer);

        VOID DeleteArray(cuda_array array);

        VOID ReadBuffer(std::string&, VOID* dst);

        VOID WriteBuffer(cuda_buffer buffer, VOID* data);
        
        VOID BindArrayToTexture(LPCSTR textur, cuda_array array, CUfilter_mode filterMode, CUaddress_mode addressMode, UINT flags);

        map_info MapGraphicsResource(cuda_buffer buffer);

        VOID UnmapGraphicsResource(cuda_buffer buffer);

        cuda_buffer GetBuffer(std::string&);

        cuda_record CreateRecord(VOID);

        cuda_record CreateRecord(cuda_stream stream);

        cuda_stream CreateStream(VOID);

        VOID CallKernel(cuda_kernel kernel);

        cuda_kernel GetKernel(LPCSTR name);

        CUcontext* GetContext(VOID);

        VOID DestroyRecord(cuda_record record);
        
        INT CheckResourceExists(std::string& name)
        {
            if(m_buffer.find(name) != m_buffer.end() || m_array.find(name) != m_array.end()) 
            {
                return TRUE;
            }
            LOG_CRITICAL_ERROR("Resource does not exists!");
            return FALSE;
        }

        INT CheckResourceNotExists(std::string& name)
        {
            if(m_buffer.find(name) == m_buffer.end() && m_array.find(name) == m_array.end()) 
            {
                return TRUE;
            }
            LOG_CRITICAL_ERROR("Resource exists!");
            return FALSE;
        }

        static UINT GetThreadCount(UINT elements, UINT blockSize)
        {
            if(elements % blockSize == 0)
            {
                return elements;
            }
            return (elements / blockSize + 1) * blockSize;
        }

        static INT GetMaxThreadsPerSM(VOID)
        {
            return 2048;//m_sProps.maxThreadsPerMultiProcessor;
        }

        static INT GetMaxBlocksPerSM(VOID)
        {
            return 8; //Fermi
        }
        
        ~cudah(VOID);
    };
}

