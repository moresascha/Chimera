#pragma once
#include "CMTypes.h"
#include "CMCommon.h"
#include "ResourceAPI.h"

namespace chimera
{
    class IDeviceTexture : public VRamHandle
    {
    public:
        virtual VOID* VGetDevicePtr(VOID) = 0;

        virtual ~IDeviceTexture(VOID) {}
    };

    class IVertexBuffer
    {

    };

    class IGeometry : public VRamHandle
    {
    public:
        virtual VOID VBind(VOID) = 0;
        
        virtual VOID VDraw(VOID) = 0;
        
        virtual VOID VDraw(UINT start, UINT count) = 0;

        virtual VOID VSetTopology(GeometryTopology topo) = 0;

        virtual VOID VSetVertexBuffer(CONST FLOAT* vertices, UINT count, UINT byteStride) = 0;

        virtual VOID VSetIndexBuffer(CONST UINT* indices, UINT size) = 0;

        virtual VOID VAddInstanceBuffer(FLOAT* vertices, UINT count, UINT byteStride) = 0;

        virtual ~IGeometry(VOID) {}
    };

    class IBlendState
    {
    public:
        virtual VOID* VGetDevicePtr(VOID) = 0;
        
        virtual ~IBlendState(VOID) {}
    };

    class IRasterState
    {
    public:
        virtual VOID* VGetDevicePtr(VOID) = 0;

        virtual ~IRasterState(VOID) {}
    };

    class IDepthStencilState
    {
    public:
        virtual VOID* VGetDevicePtr(VOID) = 0;
        
        virtual ~IDepthStencilState(VOID) {}
    };

    class IGraphicsStateFactroy
    {
    public:
        virtual IBlendState* VCreateBlendState(CONST BlendStateDesc* desc) = 0;

        virtual IRasterState* VCreateRasterState(CONST RasterStateDesc* desc) = 0;

        virtual IDepthStencilState* VCreateDepthStencilState(CONST DepthStencilStateDesc* desc) = 0;

        virtual ~IGraphicsStateFactroy(VOID) { }
    };

    class IRenderTarget
    {
    public:
        virtual BOOL VOnRestore(UINT width, UINT height, GraphicsFormat format, BOOL depthBuffer = TRUE, BOOL cubeMap = FALSE, UINT arraySize = 1) = 0;

        virtual VOID VClear(VOID) = 0;

        virtual VOID VBind(VOID) = 0;

        virtual VOID VSetClearColor(FLOAT r, FLOAT g, FLOAT b, FLOAT a) = 0;

        virtual UINT VGetWidth(VOID) = 0;

        virtual UINT VGetHeight(VOID) = 0;

        virtual IDeviceTexture* VGetTexture(VOID) = 0;

        virtual IDeviceTexture* VGetDepthStencilTexture(VOID) = 0;

        virtual ~IRenderTarget(VOID) {}
    };

    class IAlbedoBuffer
    {
    public:
        virtual VOID VClearAndBindRenderTargets(VOID) = 0;

        virtual VOID VUnbindRenderTargets(VOID) = 0;
        
        virtual VOID VOnRestore(UINT w, UINT h) = 0;

        virtual IRenderTarget* VGetRenderTarget(Diff_RenderTarget target) = 0;

        virtual IRenderTarget* VGetDepthStencilTarget(VOID) = 0;

        virtual ~IAlbedoBuffer(VOID) {}
    };

    class IConstShaderBuffer
    {
    public:
        virtual VOID VInit(UINT byteSize, VOID* data = NULL) = 0;

        virtual VOID* VMap(VOID) = 0;

        virtual VOID VUnmap(VOID) = 0;

        virtual VOID VSetData(VOID* data) = 0;

        virtual VOID VSetFromMatrix(CONST util::Mat4& mat) = 0;

        virtual VOID VActivate(ConstShaderBufferSlot slot, UINT shader = ACTIVATE_ALL) = 0;

        virtual VOID* VGetDevicePtr(VOID) = 0;

        virtual ~IConstShaderBuffer(VOID) {}
    };

    class IRenderer 
    {
    public:
        virtual BOOL VCreate(CM_WINDOW_CALLBACK cb, CM_INSTANCE instance, LPCWSTR wndTitle, UINT width, UINT height) = 0;
        virtual VOID VDestroy(VOID) = 0;

        virtual UINT VGetWidth(VOID) = 0;
        virtual UINT VGetHeight(VOID) = 0;

        virtual VOID VSetBackground(FLOAT r, FLOAT g, FLOAT b, FLOAT a) = 0;

        virtual BOOL VOnRestore(VOID) = 0;

        virtual VOID VResize(UINT w, UINT h) = 0;

        virtual VOID VPreRender(VOID) = 0;
        virtual VOID VPostRender(VOID) = 0;
        virtual VOID VPresent(VOID) = 0;

        virtual VOID VSetViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos) = 0;
        virtual VOID VSetProjectionTransform(CONST util::Mat4& mat, FLOAT distance) = 0;
        virtual VOID VSetWorldTransform(CONST util::Mat4& mat) = 0;

        virtual VOID VPushViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos) = 0;
        virtual VOID VPopViewTransform(VOID) = 0;

        virtual VOID VPushProjectionTransform(CONST util::Mat4& mat, FLOAT distance) = 0;
        virtual VOID VPopProjectionTransform(VOID) = 0;

        virtual VOID VPushWorldTransform(CONST util::Mat4& mat) = 0;
        virtual VOID VPopWorldTransform(VOID) = 0;

        virtual VOID VPushMaterial(IMaterial& mat) = 0;
        virtual VOID VPopMaterial(VOID) = 0;

        virtual VOID VSetViewPort(UINT w, UINT h) = 0;

        virtual VOID VPushBlendState(IBlendState* state) = 0;
        virtual VOID VPopBlendState(VOID) = 0;

        virtual VOID VSetDefaultMaterial(VOID) = 0;

        virtual IAlbedoBuffer* VGetAlbedoBuffer(VOID) = 0;

        virtual VOID VClearAndBindBackBuffer(VOID) = 0;
        virtual VOID VBindBackBuffer(VOID) = 0;

        virtual VOID VPushCurrentRenderTarget(IRenderTarget* target) = 0;
        virtual VOID VPopCurrentRenderTarget(VOID) = 0;
        virtual IRenderTarget* VGetCurrentRenderTarget(VOID) = 0;

        virtual VOID VPushRasterState(IRasterState* rstate) = 0;
        virtual VOID VPopRasterState(VOID) = 0;

        virtual VOID VPushDepthStencilState(IDepthStencilState* rstate, UINT stencilRef = 0) = 0;
        virtual VOID VPopDepthStencilState(VOID) = 0;

        virtual VOID VSetTexture(TextureSlot slot, IDeviceTexture* texture) = 0;
        virtual VOID VSetTextures(TextureSlot startSlot, IDeviceTexture** texture, UINT count) = 0;

        virtual VOID VSetDiffuseTexture(IDeviceTexture* texture) = 0;

        virtual VOID VSetNormalMapping(BOOL enable) = 0;

        virtual IConstShaderBuffer* VGetConstShaderBuffer(ConstShaderBufferSlot slot) = 0;

        virtual IShaderCache* VGetShaderCache(VOID) = 0;

        virtual CM_HWND VGetWindowHandle(VOID) = 0;

        virtual VOID VDrawScreenQuad(INT x, INT y, INT w, INT h) = 0;

        virtual VOID VDrawScreenQuad(VOID) = 0;

        virtual VOID VDrawLine(INT x, INT y, INT w, INT h) = 0;

        virtual ~IRenderer(VOID) {}
    };

    class IShader
    {
    public:
        virtual BOOL VCompile(ErrorLog* log = NULL) = 0;

        virtual VOID VBind(VOID) = 0;

        virtual VOID VUnbind(VOID) = 0;

        virtual ShaderType VGetType(VOID) = 0;

        virtual ~IShader(VOID) {}
    };

    class IShaderProgram
    {
    public:
        virtual VOID VBind(VOID) = 0;

        virtual VOID VUnbind(VOID) = 0;

        virtual VOID VAddShader(IShader* shader) = 0;

        virtual BOOL VCompile(ErrorLog* log = NULL) = 0;

        virtual ~IShaderProgram(VOID) {}
    };

    class IShaderCache
    {
    public:
        virtual IShader* VGetVertexShader(LPCSTR name) = 0;

        virtual IShader* VGetFragmentShader(LPCSTR name) = 0;

        virtual IShader* VGetGeometryShader(LPCSTR name) = 0;

        virtual IShaderProgram* VGetShaderProgram(LPCSTR name) = 0;

        virtual IShader* VCreateShader(LPCSTR name, CONST CMShaderDescription* desc, ShaderType t) = 0;

        virtual IShaderProgram* VCreateShaderProgram(LPCSTR name, CONST CMShaderProgramDescription* desc) = 0;

        virtual ~IShaderCache(VOID) {}
    };

    class IPicker
    {
    public:
        virtual BOOL VCreate(VOID) = 0;

        virtual VOID VPostRender(VOID) = 0;

        virtual VOID VRender(VOID) = 0;

        virtual BOOL VHasPicked(VOID) CONST  = 0;

        virtual ActorId VPick(VOID) CONST = 0;

        virtual ~IPicker(VOID) {}
    };

    //factory

    class IShaderFactory
    {
    public:
        virtual IShaderProgram* VCreateShaderProgram(VOID) = 0;

        virtual IShaderProgram* VCreateShaderProgram(CONST CMShaderProgramDescription* desc)
        {
            IShaderProgram* p = VCreateShaderProgram();

            if(desc->vs.function)
            {
                p->VAddShader(VCreateVertexShader(&desc->vs));
            }

            if(desc->fs.function)
            {
                p->VAddShader(VCreateFragmentShader(&desc->fs));
            }

            if(desc->gs.function)
            {
                p->VAddShader(VCreateGeometryShader(&desc->gs));
            }
            return p;
        }

        virtual IShader* VCreateVertexShader(CONST CMVertexShaderDescription* desc) = 0;

        virtual IShader* VCreateFragmentShader(CONST CMShaderDescription* desc) = 0;

        virtual IShader* VCreateGeometryShader(CONST CMShaderDescription* desc) = 0;

        virtual ~IShaderFactory(VOID) {}
        //TODO: tessellation
    };

    class IGraphicsFactory
    {
    public:
        virtual std::unique_ptr<IRenderer> VCreateRenderer(VOID) = 0;

        virtual std::unique_ptr<IRenderScreen> VCreateRenderScreen(VOID) = 0;

        virtual std::unique_ptr<IRenderTarget> VCreateRenderTarget(VOID) = 0;

        virtual std::unique_ptr<IGeometry> VCreateGeoemtry(VOID) = 0;

        virtual std::unique_ptr<IDeviceTexture> VCreateTexture(CONST CMTextureDescription* desc) = 0;

        virtual std::unique_ptr<IConstShaderBuffer> VCreateConstShaderBuffer(VOID) = 0;

        virtual std::unique_ptr<IGraphicsStateFactroy> VCreateStateFactory(VOID) = 0;

        virtual std::unique_ptr<IShaderFactory> VCreateShaderFactory(VOID)  = 0;

        virtual ~IGraphicsFactory(VOID) {}
    };
}
