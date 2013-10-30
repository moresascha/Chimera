#pragma once
#include "CMCommon.h"

namespace chimera
{
    class _ActorDescription;
    class ActorComponent;
    class IProcess;
    class IEvent;
    class CMResource;
    class Command;
    class IActor;
    class IGameView;
    class IMaterial;
    class ParticleSystem;
    class Frustum;
    class ILogic;
    class IEventManager;
    class IResourceCache;
    class IVRamManager;
    class ITimer;
    class IScreenElement;
    class IRenderScreen;
    class IScriptEventManager;
    class IHumanView;
    class IInputHandler;
    class IFileSystem;
    class IConfig;
    class IRenderer;
    class IPicker;
    class ISoundEngine;
    class ISoundSystem;
    class IRenderTarget;
    class IGeometry;
    class IMesh;
    class IScript;
    class IFontManager;
    class ISceneNode;
    class ISceneGraph;
    class ICamera;
    class IKeyListener;
    class IMouseListener;
    class IProcessManager;
    class ISoundFactory;
    class IPhysicsSystem;
    class IResourceFile;
    class IResourceLoader;
    class IResourceDecompressor;
    class IResHandle;
    class IExtraRessourceData;
    class CMIStream;
    class ActorDescription;
    class ILevel;
    class ICommandInterpreter;
    class IVRamHandleCreator;
    class IScheduler;
    class IGraphicsSettings;
    class IShaderCache;
    class IActorComponent;
    class IActorFactory;
    class MaterialSet;
    class IEffect;
    class IEffectChain;
    class IEffectFactory;
    class IEffectFactoryFactory;
    class IEnvironmentLighting;

    namespace util
    {
        class Vec3;
        class Vec4;
        class Mat4;
        class AxisAlignedBB;
    };

    enum ViewType 
    {
        eViewType_Human,
        eProjectionType_Controller,
        eProjectionType_AI,
        eProjectionType_Other
    };

    enum ErrorCode
    {
        eErrorCode_Success,
        eErrorCode_InvalidValue
    };

    typedef fastdelegate::FastDelegate0<VOID> UpdateAction;
    typedef UINT RenderPath;
    typedef fastdelegate::FastDelegate3<INT, INT, INT> MouseScrollAction;
    typedef ISceneNode* (*SceneNodeCreator)(IHumanView*, IActor*);
    typedef size_t FactoryPtr;
    typedef WNDPROC CM_WINDOW_CALLBACK;
    typedef HINSTANCE CM_INSTANCE;
    typedef HWND CM_HWND;
    typedef UINT GameViewId;
    typedef util::Vec4 Color;
    typedef ULONG ActorId;
    typedef UINT ComponentId;
    typedef fastdelegate::FastDelegate0<> OnFileChangedCallback;
    typedef fastdelegate::FastDelegate1<ULONG> _FDelegate;
    typedef IActorComponent* (*ActorComponentCreator)(VOID);
    typedef ULONG EventType;
    typedef std::shared_ptr<IEvent> IEventPtr;
    typedef fastdelegate::FastDelegate1<IEventPtr> EventListener;
    typedef CMResource VRamResource;
    typedef fastdelegate::FastDelegate1<Command&, BOOL> CommandHandler;
    typedef std::string ErrorLog;
    typedef INT PixelFormat;
    typedef INT GameState;

    struct FLOAT2
    {
        FLOAT x;
        FLOAT y;
    };

    enum GraphicsFormat
    {
        eFormat_UNKNOWN = 0, 
        eFormat_R32G32B32A32_TYPELESS = 1, 
        eFormat_R32G32B32A32_FLOAT = 2, 
        eFormat_R32G32B32A32_UINT = 3, 
        eFormat_R32G32B32A32_SINT = 4, 
        eFormat_R32G32B32_TYPELESS = 5, 
        eFormat_R32G32B32_FLOAT = 6, 
        eFormat_R32G32B32_UINT = 7, 
        eFormat_R32G32B32_SINT = 8, 
        eFormat_R16G16B16A16_TYPELESS = 9, 
        eFormat_R16G16B16A16_FLOAT = 10, 
        eFormat_R16G16B16A16_UNORM = 11, 
        eFormat_R16G16B16A16_UINT = 12, 
        eFormat_R16G16B16A16_SNORM = 13, 
        eFormat_R16G16B16A16_SINT = 14, 
        eFormat_R32G32_TYPELESS = 15, 
        eFormat_R32G32_FLOAT = 16, 
        eFormat_R32G32_UINT = 17, 
        eFormat_R32G32_SINT = 18, 
        eFormat_R32G8X24_TYPELESS = 19, 
        eFormat_D32_FLOAT_S8X24_UINT = 20, 
        eFormat_R32_FLOAT_X8X24_TYPELESS = 21, 
        eFormat_X32_TYPELESS_G8X24_UINT = 22, 
        eFormat_R10G10B10A2_TYPELESS = 23, 
        eFormat_R10G10B10A2_UNORM = 24, 
        eFormat_R10G10B10A2_UINT = 25, 
        eFormat_R11G11B10_FLOAT = 26, 
        eFormat_R8G8B8A8_TYPELESS = 27, 
        eFormat_R8G8B8A8_UNORM = 28, 
        eFormat_R8G8B8A8_UNORM_SRGB = 29, 
        eFormat_R8G8B8A8_UINT = 30, 
        eFormat_R8G8B8A8_SNORM = 31, 
        eFormat_R8G8B8A8_SINT = 32, 
        eFormat_R16G16_TYPELESS = 33, 
        eFormat_R16G16_FLOAT = 34, 
        eFormat_R16G16_UNORM = 35, 
        eFormat_R16G16_UINT = 36, 
        eFormat_R16G16_SNORM = 37, 
        eFormat_R16G16_SINT = 38, 
        eFormat_R32_TYPELESS = 39, 
        eFormat_D32_FLOAT = 40, 
        eFormat_R32_FLOAT = 41, 
        eFormat_R32_UINT = 42, 
        eFormat_R32_SINT = 43, 
        eFormat_R24G8_TYPELESS = 44, 
        eFormat_D24_UNORM_S8_UINT = 45, 
        eFormat_R24_UNORM_X8_TYPELESS = 46, 
        eFormat_X24_TYPELESS_G8_UINT = 47, 
        eFormat_R8G8_TYPELESS = 48, 
        eFormat_R8G8_UNORM = 49, 
        eFormat_R8G8_UINT = 50, 
        eFormat_R8G8_SNORM = 51, 
        eFormat_R8G8_SINT = 52, 
        eFormat_R16_TYPELESS = 53, 
        eFormat_R16_FLOAT = 54, 
        eFormat_D16_UNORM = 55, 
        eFormat_R16_UNORM = 56, 
        eFormat_R16_UINT = 57, 
        eFormat_R16_SNORM = 58, 
        eFormat_R16_SINT = 59, 
        eFormat_R8_TYPELESS = 60, 
        eFormat_R8_UNORM = 61, 
        eFormat_R8_UINT = 62, 
        eFormat_R8_SNORM = 63, 
        eFormat_R8_SINT = 64, 
        eFormat_A8_UNORM = 65, 
        eFormat_R1_UNORM = 66, 
        eFormat_R9G9B9E5_SHAREDEXP = 67, 
        eFormat_R8G8_B8G8_UNORM = 68, 
        eFormat_G8R8_G8B8_UNORM = 69, 
        eFormat_BC1_TYPELESS = 70, 
        eFormat_BC1_UNORM = 71, 
        eFormat_BC1_UNORM_SRGB = 72, 
        eFormat_BC2_TYPELESS = 73, 
        eFormat_BC2_UNORM = 74, 
        eFormat_BC2_UNORM_SRGB = 75, 
        eFormat_BC3_TYPELESS = 76, 
        eFormat_BC3_UNORM = 77, 
        eFormat_BC3_UNORM_SRGB = 78, 
        eFormat_BC4_TYPELESS = 79, 
        eFormat_BC4_UNORM = 80, 
        eFormat_BC4_SNORM = 81, 
        eFormat_BC5_TYPELESS = 82, 
        eFormat_BC5_UNORM = 83, 
        eFormat_BC5_SNORM = 84, 
        eFormat_B5G6R5_UNORM = 85, 
        eFormat_B5G5R5A1_UNORM = 86, 
        eFormat_B8G8R8A8_UNORM = 87, 
        eFormat_B8G8R8X8_UNORM = 88, 
        eFormat_R10G10B10_XR_BIAS_A2_UNORM = 89, 
        eFormat_B8G8R8A8_TYPELESS = 90, 
        eFormat_B8G8R8A8_UNORM_SRGB = 91, 
        eFormat_B8G8R8X8_TYPELESS = 92, 
        eFormat_B8G8R8X8_UNORM_SRGB = 93, 
        eFormat_BC6H_TYPELESS = 94, 
        eFormat_BC6H_UF16 = 95, 
        eFormat_BC6H_SF16 = 96, 
        eFormat_BC7_TYPELESS = 97, 
        eFormat_BC7_UNORM = 98, 
        eFormat_BC7_UNORM_SRGB = 99, 
        eFormat_AYUV = 100, 
        eFormat_Y410 = 101, 
        eFormat_Y416 = 102, 
        eFormat_NV12 = 103, 
        eFormat_P010 = 104, 
        eFormat_P016 = 105, 
        eFormat_420_OPAQUE = 106, 
        eFormat_YUY2 = 107, 
        eFormat_Y210 = 108, 
        eFormat_Y216 = 109, 
        eFormat_NV11 = 110, 
        eFormat_AI44 = 111, 
        eFormat_IA44 = 112, 
        eFormat_P8 = 113, 
        eFormat_A8P8 = 114, 
        eFormat_B4G4R4A4_UNORM = 115, 
        eFormat_FORCE_UINT = 116,
    };

    enum ConstShaderBufferSlot 
    {
        eViewBuffer = 0,
        eProjectionBuffer = 1,
        eModelBuffer = 2,
        eMaterialBuffer = 3,
        eCubeMapViewsBuffer = 4,
        eLightingBuffer = 5,
        eFontBuffer = 6,
        eBoundingGeoBuffer = 7,
        eActorIdBuffer = 8,
        eSelectedActorIdBuffer = 9,
        eGuiColorBuffer = 10,
        eHasNormalMapBuffer = 11,
        eEnvLightingBuffer = 12,
        BufferCnt
    };

    enum TextureSlot
    {
        eDiffuseColorSampler,
        eWorldPositionSampler,
        eNormalsSampler,
        eDiffuseMaterialSpecRSampler,
        eAmbientMaterialSpecGSampler,
        eDiffuseColorSpecBSampler,
        ePointLightShadowCubeMapSampler,
        eGuiSampler,
        eNormalColorSampler,
        eSceneSampler,
        eEffect0,
        eEffect1,
        eEffect2,
        eEffect3,
        SamplerCnt
    };

    enum Diff_RenderTarget
    {
        eDiff_WorldPositionTarget,
        eDiff_NormalsTarget,
        eDiff_DiffuseMaterialSpecRTarget,
        eDiff_AmbientMaterialSpecGTarget,
        eDiff_DiffuseColorSpecBTarget,
        eDiff_ReflectionStrTarget,
        Diff_SamplersCnt
    };

    enum GeometryTopology
    {
        eTopo_Triangles,
        eTopo_TriangleStrip,
        eTopo_Lines,
        eTopo_LineStrip,
        eTopo_Points
    };

    enum Fillmode
    {
        eFillMode_Wire = 2,
        eFillMode_Solid = 3
    };

    enum CullMode
    {
        eCullMode_None = 1,
        eCullMode_Front = 2,
        eCullMode_Back = 3
    };

    struct RasterStateDesc
    {
        Fillmode FillMode;
        CullMode CullMode;
        BOOL FrontCounterClockwise;
        INT DepthBias;
        FLOAT DepthBiasClamp;
        FLOAT SlopeScaledDepthBias;
        BOOL DepthClipEnable;
        BOOL ScissorEnable;
        BOOL MultisampleEnable;
        BOOL AntialiasedLineEnable;
    };

    enum Blend
    {
        eBlend_Zero = 1,
        eBlend_One = 2,
        eBlend_SrcColor = 3,
        eBlend_InvSrcColor = 4,
        eBlend_SrcAlpha = 5
    };

    enum BlendOP
    {
        eBlendOP_Add = 1,
        eBlendOP_Sub = 2,
        eBlendOP_Rev_Sub = 3,
        eBlendOP_Min = 4,
        eBlendOP_Max = 5
    };

    struct RenderTargetBlendDesc
    {
        BOOL BlendEnable;
        Blend SrcBlend;
        Blend DestBlend;
        BlendOP BlendOp;
        Blend SrcBlendAlpha;
        Blend DestBlendAlpha;
        BlendOP BlendOpAlpha;
        UINT8 RenderTargetWriteMask;
    };

    struct BlendStateDesc
    {
        BOOL AlphaToCoverageEnable;
        BOOL IndependentBlendEnable;
        RenderTargetBlendDesc RenderTarget[8];
    } ;

    enum DepthWriteMask
    {
        eDepthWriteMask_Zero = 0,
        eDepthWriteMask_All = 1
    };

    enum ColorWriteMask
    {
        eColorWriteRed    = 1,
        eColorWriteGreen    = 2,
        eColorWriteBlue    = 4,
        eColorWriteAlpha    = 8,
        eColorWriteAll    = eColorWriteRed | eColorWriteGreen | eColorWriteBlue | eColorWriteAlpha 
    };

    enum ComparisonFunc
    {
        eCompareFunc_Never = 1,
        eCompareFunc_Less = 2,
        eCompareFunc_Equal = 3,
        eCompareFunc_Less_Equal = 4,
        eCompareFunc_Greater = 5,
        eCompareFunc_Not_Equal = 6,
        eCompareFunc_Greater_Equal = 7,
        eCompareFunc_Always = 8
    };

    enum StencilOperation
    {
        eStencilOP_Keep = 1,
        eStencilOP_Zero = 2,
        eStencilOP_Replace = 3,
        eStencilOP_Incr_Sat = 4,
        eStencilOP_Decr_Sat = 5,
        eStencilOP_Invert = 6,
        eStencilOP_Incr = 7,
        eStencilOP_Decr = 8
    };

    struct DepthStencilOperationDesc
    {
        StencilOperation StencilFailOp;
        StencilOperation StencilDepthFailOp;
        StencilOperation StencilPassOp;
        ComparisonFunc StencilFunc;
    } ;

    struct DepthStencilStateDesc
    {
        BOOL DepthEnable;
        DepthWriteMask DepthWriteMask;
        ComparisonFunc DepthFunc;
        BOOL StencilEnable;
        UINT8 StencilReadMask;
        UINT8 StencilWriteMask;
        DepthStencilOperationDesc FrontFace;
        DepthStencilOperationDesc BackFace;
    };

    enum ProcessType
    {
        eProcessType_Normal,
        eProcessType_Realtime,
        eProcessType_Actor_Realtime
    };

    enum ProcessState 
    {
        eProcessState_Uninitialized,
        eProcessState_Removed,
        eProcessState_Running,
        eProcessState_Paused,
        eProcessState_Succed,
        eProcessState_Failed,
        eProcessState_Aborted
    };

    enum GraphicsSettingType
    {
        eGraphicsSetting_Albedo = 0,
        eGraphicsSetting_Lighting = 1,
    };

    enum ShaderType
    {
        eShaderType_VertexShader,
        eShaderType_FragmentShader,
        eShaderType_GeometryShader
    };

    enum SaveLevelFormat
    {
        eXML,
        eCNT
    };

    enum TextureMiscFlags
    {
        eTextureMiscFlags_GenerateMipMaps
    };

    struct CMFontMetrics
    {
        FLOAT leftU;
        FLOAT rightU;
        UINT pixelWidth;
    };

    struct CMCharMetric
    {
        UCHAR id;
        UINT x;
        UINT y;
        UINT width;
        UINT height;
        INT xoffset;
        INT yoffset;
        UINT xadvance;
    };

    struct CMFontStyle
    {
        BOOL italic;
        BOOL bold;
        UINT charCount;
        UINT lineHeight;
        UINT texWidth;
        UINT texHeight;
        UINT size;
        UINT base;
        std::string textureFile;
        std::string metricFile;
        std::string name;
    };

    struct CMTextureDescription
    {
        UINT width;
        UINT height;
        UINT mipmapLevels;
        UINT arraySize;
        UINT usage;
        UINT cpuAccess;
        GraphicsFormat format;
        TextureMiscFlags miscflags;
        VOID* data;
    };

    struct CMShaderDescription
    {
        CMShaderDescription(VOID) : file(NULL), function(NULL)
        {

        }
        LPCTSTR file; 
        LPCSTR function;
    };

    struct CMVertexInputLayout
    {
        LPCSTR name;
        UINT position;
        UINT slot;
        GraphicsFormat format;
        BOOL instanced;
    };

    struct CMVertexShaderDescription : public CMShaderDescription
    {
        CMVertexInputLayout inputLayout[16];
        UINT layoutCount;
    };

    struct CMShaderProgramDescription
    {
        CMVertexShaderDescription vs;
        CMShaderDescription fs;
        CMShaderDescription gs;
        /*CMShaderDescription ts;
        CMShaderDescription cs; */
    };

    struct CMDimension
    {
        UINT x;
        UINT y;
        UINT w;
        UINT h;

        CMDimension(VOID)
        {
            x = 0;
            y = 0;
            w = 0;
            h = 0;
        }

        CMDimension(CONST CMDimension& dim)
        {
            x = dim.x;
            y = dim.y;
            w = dim.w;
            h = dim.h;
        }
    };

    enum CommandArgType
    {
        eCommandArgument_Float = 1,
        eCommandArgument_Int = 2,
        eCommandArgument_String = 3
    };

    enum ProjectionType
    {
        eProjectionType_Perspective,
        eProjectionType_Orthographic,
        eProjectionType_OrthographicOffCenter
    };

    class CMResource
    {

    public:
        std::string m_name;
        CMResource(CONST std::string &name) : m_name(name) 
        {
            std::transform(m_name.begin(), m_name.end(), m_name.begin(), ::tolower);
        }

        CMResource(LPCSTR name)
        {
            m_name = name;
            std::transform(m_name.begin(), m_name.end(), m_name.begin(), ::tolower);
        }

        CMResource(CONST CMResource& r)
        {
            this->m_name = r.m_name;
        }

        CMResource(VOID) : m_name("unknown") {}

        VOID CMResource::operator=(std::string& str) 
        {
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);
            this->m_name = str;
        }

        VOID CMResource::operator=(CHAR* chars) 
        {
            std::string str(chars);
            std::transform(str.begin(), str.end(), str.begin(), ::tolower);
            this->m_name = str;
        }

        VOID CMResource::operator=(CONST CMResource& res)
        {
            m_name = res.m_name;
        }

        BOOL CMResource::operator!=(CONST CMResource& res)
        {
            return m_name != res.m_name;
        }

        BOOL CMResource::operator==(CONST CMResource& res)
        {
            return m_name == res.m_name;
        }

        ~CMResource(VOID) { }
    };

    template <typename T>
    T* FindFactory(FactoryPtr* facts, UINT type, size_t* size)
    {
        for(UINT i = 0; CM_FACTORY_END != facts[0]; i+=3)
        {
            if(type == facts[i])
            {
                if(size)
                {
                    *size = (size_t)facts[i+2];
                }
                return (T*)(facts[i+1]);
            }
        }
        return (T*)(NULL);
    }

    template <typename T>
    T* FindFactory(FactoryPtr* facts, UINT type)
    {
        return FindFactory<T>(facts, type, NULL);
    }

    template <typename T>
    T* CopyFactory(T* fact, size_t size)
    {
        VOID* mem = malloc(size);
        memcpy(mem, fact, size);
        return (T*)(mem);
    }

    template <typename T>
    T* FindAndCopyFactory(FactoryPtr* facts, UINT type)
    {
        size_t size;
        T* fact = FindFactory<T>(facts, type, &size);
        return CopyFactory<T>(fact, size);
    }
};

namespace tinyxml2
{
    class XMLElement;
    class XMLDocument;
};
