#pragma once
#include "CMTypes.h"
#include "CMCommon.h"
#include "ResourceAPI.h"

namespace chimera
{
    class IDeviceTexture : public VRamHandle
    {
    public:
        virtual void* VGetDevicePtr(void) = 0;

        virtual void* VGetViewDevicePtr(void) = 0;

        virtual ~IDeviceTexture(void) {}
    };

    class IDeviceBuffer
    {
    public:
        virtual void VSetData(const void* v, uint bytes) = 0;

        virtual void VCreate(void) = 0;

        virtual void VBind(void) = 0;

        virtual uint VGetByteCount(void) const = 0;

        virtual uint VGetElementCount(void) const = 0;

        virtual void* VGetDevicePtr(void) = 0;

        virtual ~IDeviceBuffer(void) {}
    };

    class IVertexBuffer : public virtual IDeviceBuffer
    {
    public:
        virtual uint VGetStride(void) const = 0;

        virtual void VInitParamater(uint vertexCount, uint stride, const void* data = NULL, bool cpuAccessFlags = false) = 0;

        virtual uint VGetOffset(void) const = 0;

        virtual ~IVertexBuffer(void) {}
    };

    class IGeometry : public VRamHandle
    {
    public:
        virtual void VBind(void) = 0;
        
        virtual void VDraw(void) = 0;
        
        virtual void VDraw(uint start, uint count) = 0;

        virtual void VSetTopology(GeometryTopology topo) = 0;

        virtual void VSetVertexBuffer(const float* vertices, uint count, uint byteStride, bool cpuWrite = false) = 0;

        virtual void VSetIndexBuffer(const uint* indices, uint size) = 0;

        //virtual VOID VAddInstanceBuffer(FLOAT* vertices, UINT count, UINT byteStride) = 0;

        virtual void VSetInstanceBuffer(IVertexBuffer* instanceBuffer) = 0;

        virtual IVertexBuffer* VGetVertexBuffer(void) = 0;

        virtual IVertexBuffer* VGetInstanceBuffer(void) = 0;

        virtual IDeviceBuffer* VGetIndexBuffer(void) = 0;

        virtual ~IGeometry(void) {}
    };

    class IBlendState
    {
    public:
        virtual void* VGetDevicePtr(void) = 0;
        
        virtual ~IBlendState(void) {}
    };

    class IRasterState
    {
    public:
        virtual void* VGetDevicePtr(void) = 0;

        virtual ~IRasterState(void) {}
    };

    class IDepthStencilState
    {
    public:
        virtual void* VGetDevicePtr(void) = 0;
        
        virtual ~IDepthStencilState(void) {}
    };

    class IGraphicsStateFactroy
    {
    public:
        virtual IBlendState* VCreateBlendState(const BlendStateDesc* desc) = 0;

        virtual IRasterState* VCreateRasterState(const RasterStateDesc* desc) = 0;

        virtual IDepthStencilState* VCreateDepthStencilState(const DepthStencilStateDesc* desc) = 0;

        virtual ~IGraphicsStateFactroy(void) { }
    };

    class IRenderTarget
    {
    public:
        virtual bool VOnRestore(uint width, uint height, GraphicsFormat format, bool depthBuffer = true, bool cubeMap = false, uint arraySize = 1) = 0;

        virtual void VClear(void) = 0;

        virtual void VBind(void) = 0;

        virtual void VSetClearColor(float r, float g, float b, float a) = 0;

        virtual uint VGetWidth(void) = 0;

        virtual uint VGetHeight(void) = 0;

        virtual IDeviceTexture* VGetTexture(void) = 0;

        virtual IDeviceTexture* VGetDepthStencilTexture(void) = 0;

        virtual ~IRenderTarget(void) {}
    };

    class IAlbedoBuffer
    {
    public:
        virtual void VClearAndBindRenderTargets(void) = 0;

        virtual void VUnbindRenderTargets(void) = 0;
        
        virtual void VOnRestore(uint w, uint h) = 0;

        virtual IRenderTarget* VGetRenderTarget(Diff_RenderTarget target) = 0;

        virtual IRenderTarget* VGetDepthStencilTarget(void) = 0;

        virtual ~IAlbedoBuffer(void) {}
    };

    class IConstShaderBuffer
    {
    public:
        virtual void VInit(uint byteSize, void* data = NULL) = 0;

        virtual void* VMap(void) = 0;

        virtual void VUnmap(void) = 0;

        virtual void VSetData(void* data) = 0;

        virtual void VSetFromMatrix(const util::Mat4& mat) = 0;

        virtual void VActivate(ConstShaderBufferSlot slot, uint shader = ACTIVATE_ALL) = 0;

        virtual void* VGetDevicePtr(void) = 0;

        virtual ~IConstShaderBuffer(void) {}
    };

    class IRenderer 
    {
    public:
        virtual bool VCreate(CM_WINDOW_CALLBACK cb, CM_INSTANCE instance, LPCWSTR wndTitle, uint width, uint height) = 0;
        virtual void VDestroy(void) = 0;

        virtual uint VGetWidth(void) = 0;
        virtual uint VGetHeight(void) = 0;

        virtual void VSetBackground(float r, float g, float b, float a) = 0;

        virtual bool VOnRestore(void) = 0;

        virtual void VResize(uint w, uint h) = 0;

        virtual void VPreRender(void) = 0;
        virtual void VPostRender(void) = 0;
        virtual void VPresent(void) = 0;

        virtual void VSetViewTransform(const util::Mat4& mat, const util::Mat4& invMat, const util::Vec3& eyePos) = 0;
        virtual void VSetProjectionTransform(const util::Mat4& mat, float distance) = 0;
        virtual void VSetWorldTransform(const util::Mat4& mat) = 0;

        virtual void VPushViewTransform(const util::Mat4& mat, const util::Mat4& invMat, const util::Vec3& eyePos) = 0;
        virtual void VPopViewTransform(void) = 0;

        virtual void VPushProjectionTransform(const util::Mat4& mat, float distance) = 0;
        virtual void VPopProjectionTransform(void) = 0;

        virtual void VPushWorldTransform(const util::Mat4& mat) = 0;
        virtual void VPopWorldTransform(void) = 0;

        virtual void VPushMaterial(IMaterial& mat) = 0;
        virtual void VPopMaterial(void) = 0;

        virtual void VSetViewPort(uint w, uint h) = 0;

        virtual void VPushBlendState(IBlendState* state) = 0;
        virtual void VPopBlendState(void) = 0;

        virtual void VSetDefaultMaterial(void) = 0;
        virtual void VSetDefaultTexture(void) = 0;

        virtual IAlbedoBuffer* VGetAlbedoBuffer(void) = 0;

        virtual void VPushAlphaBlendState(void) = 0;

        virtual void VClearAndBindBackBuffer(void) = 0;
        virtual void VBindBackBuffer(void) = 0;

        virtual void VPushCurrentRenderTarget(IRenderTarget* target) = 0;
        virtual void VPopCurrentRenderTarget(void) = 0;
        virtual IRenderTarget* VGetCurrentRenderTarget(void) = 0;

        virtual void VPushRasterState(IRasterState* rstate) = 0;
        virtual void VPopRasterState(void) = 0;

        virtual void VPushDepthStencilState(IDepthStencilState* rstate, uint stencilRef = 0) = 0;
        virtual void VPopDepthStencilState(void) = 0;

        virtual void VSetTexture(TextureSlot slot, IDeviceTexture* texture) = 0;
        virtual void VSetTextures(TextureSlot startSlot, IDeviceTexture** texture, uint count) = 0;

        virtual void VSetDiffuseTexture(IDeviceTexture* texture) = 0;

        virtual void VSetNormalMapping(bool enable) = 0;

        virtual IConstShaderBuffer* VGetConstShaderBuffer(ConstShaderBufferSlot slot) = 0;

        virtual IShaderCache* VGetShaderCache(void) = 0;

        virtual CM_HWND VGetWindowHandle(void) = 0;

        virtual void VSetActorId(ActorId id) = 0;

        virtual void VDrawScreenQuad(int x, int y, int w, int h) = 0;

        virtual void VDrawScreenQuad(void) = 0;

        virtual void VDrawLine(int x, int y, int w, int h) = 0;

        virtual void* VGetDevice(void) = 0;

        virtual void* VGetContext(void) = 0;

        virtual void VSetFullscreen(bool fullscreen) = 0;

        virtual void VSetLightSettings(const util::Vec4& color, const util::Vec3& position, const util::Vec3& viewDir, float radius, float angel, float intensity, bool castShadow) = 0;

        virtual void VSetLightSettings(const util::Vec4& color, const util::Vec3& position, float radius, bool castShadow) = 0;

        virtual ~IRenderer(void) {}
    };

    class IShader
    {
    public:
        virtual bool VCompile(ErrorLog* log = NULL) = 0;

        virtual void VBind(void) = 0;

        virtual void VUnbind(void) = 0;

        virtual ShaderType VGetType(void) = 0;

        virtual ~IShader(void) {}
    };

    class IShaderProgram
    {
    public:
        virtual void VBind(void) = 0;

        virtual void VUnbind(void) = 0;

        virtual void VAddShader(IShader* shader) = 0;

        virtual bool VCompile(ErrorLog* log = NULL) = 0;

        virtual ~IShaderProgram(void) {}
    };

    class IShaderCache
    {
    public:
        virtual IShader* VGetVertexShader(LPCSTR name) = 0;

        virtual IShader* VGetFragmentShader(LPCSTR name) = 0;

        virtual IShader* VGetGeometryShader(LPCSTR name) = 0;

        virtual IShaderProgram* VGetShaderProgram(LPCSTR name) = 0;

        virtual IShader* VCreateShader(LPCSTR name, const CMShaderDescription* desc, ShaderType t) = 0;

        virtual IShaderProgram* VCreateShaderProgram(LPCSTR name, const CMShaderProgramDescription* desc) = 0;

        virtual ~IShaderCache(void) {}
    };

    class IPicker
    {
    public:
        virtual bool VHasPicked(void) const  = 0;

        virtual ActorId VPick(void) = 0;

        virtual bool VOnRestore(void) = 0;

        virtual ~IPicker(void) {}
    };

    //factory

    class IShaderFactory
    {
    public:
        virtual IShaderProgram* VCreateShaderProgram(void) = 0;

        virtual IShaderProgram* VCreateShaderProgram(const CMShaderProgramDescription* desc)
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

        virtual IShader* VCreateVertexShader(const CMVertexShaderDescription* desc) = 0;

        virtual IShader* VCreateFragmentShader(const CMShaderDescription* desc) = 0;

        virtual IShader* VCreateGeometryShader(const CMShaderDescription* desc) = 0;

        virtual ~IShaderFactory(void) {}
        //TODO: tessellation
    };

    class IGraphicsFactory
    {
    public:
        virtual std::unique_ptr<IRenderer> VCreateRenderer(void) = 0;

        virtual std::unique_ptr<IRenderScreen> VCreateRenderScreen(void) = 0;

        virtual std::unique_ptr<IRenderTarget> VCreateRenderTarget(void) = 0;

        virtual std::unique_ptr<IGeometry> VCreateGeoemtry(void) = 0;

        virtual std::unique_ptr<IVertexBuffer> VCreateVertexBuffer(void) = 0;

        virtual std::unique_ptr<IDeviceBuffer> VCreateIndexBuffer(void) = 0;

        virtual std::unique_ptr<IDeviceTexture> VCreateTexture(const CMTextureDescription* desc) = 0;

        virtual std::unique_ptr<IConstShaderBuffer> VCreateConstShaderBuffer(void) = 0;

        virtual std::unique_ptr<IGraphicsStateFactroy> VCreateStateFactory(void) = 0;

        virtual std::unique_ptr<IShaderFactory> VCreateShaderFactory(void)  = 0;

        virtual ~IGraphicsFactory(void) {}
    };
}
