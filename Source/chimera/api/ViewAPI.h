#pragma once
#include "CMTypes.h"
#include "InputAPI.h"
namespace chimera
{
    class IView
    {
    protected:
        ViewId m_id;
        IActor* m_actor;
        std::string m_name;
    public:
        IView(void) : m_id((ViewId)-1), m_actor(NULL), m_name("undefined")
        {

        }

        IActor* VGetTarget(void)
        {
            return m_actor;
        }

        virtual void VOnAttach(uint viewId, IActor* actor) 
        {
            m_id = viewId;
            VSetTarget(actor);
        }

        virtual void VSetTarget(IActor* actor)
        {
            if(actor)
            {
                m_actor = actor;
            }
        }

        ViewId GetId(void) const { return m_id; }

        const std::string& GetName(void) const { return m_name; }

        void SetName(const std::string& name) { m_name = name; }

        virtual bool VOnRestore(void) { return true; };

        virtual bool VInitialise(FactoryPtr* facts) = 0;

        virtual void VOnUpdate(ulong deltaMillis) = 0;

        virtual ViewType VGetType(void) const = 0;

        virtual ~IView(void) {}
    };

    class IHumanView : public IView
    {
    public:
        virtual IRenderer* VGetRenderer(void) = 0;

        virtual IPicker* VGetPicker(void) = 0;

        virtual IFontManager* VGetFontManager(void) = 0;

        virtual ISoundEngine* VGetSoundEngine(void) = 0;

        virtual ISoundSystem* VGetSoundSystem(void) = 0;

        virtual void VAddScreenElement(std::unique_ptr<IScreenElement> element) = 0;

        virtual IGuiFactory* VGetGuiFactory(void) = 0;

        virtual ISceneGraph* VGetSceneGraph(void) = 0;

        virtual void VOnRender(void) = 0;

        virtual IGraphicsFactory* VGetGraphicsFactory(void) = 0;

        virtual void VAddScene(std::unique_ptr<IRenderScreen> screen) = 0;

        virtual void VAddSceneNodeCreator(SceneNodeCreator nc, ComponentId cmpid) = 0;

        virtual void VRemoveSceneNodeCreator(ComponentId cmpid) = 0;

        virtual void VActivateScene(LPCSTR name) = 0;

        virtual IRenderScreen* VGetSceneByName(LPCSTR name) = 0;

        virtual IScreenElement* VGetScreenElementByName(LPCSTR name) = 0;

        virtual void VOnResize(uint w, uint h) = 0;

        virtual void VSetFullscreen(bool fullscreen) = 0;

        virtual bool VIsFullscreen(void) = 0;

        virtual IEffectFactory* VGetEffectFactory(void) = 0;

        virtual IVRamManager* VGetVRamManager(void) = 0;
    };

    class ISceneGraph
    {
    public:
        virtual void VAddNode(ActorId actorid, std::unique_ptr<ISceneNode> node) = 0;

        virtual void VRemoveNode(ActorId actorid) = 0;

        virtual bool VOnRender(RenderPath path) = 0;

        virtual bool VOnRestore(void) = 0;

        virtual bool VOnUpdate(ulong millis) = 0;

        virtual std::shared_ptr<ICamera> VGetCamera(void) = 0;

        virtual void VSetCamera(std::shared_ptr<ICamera> camera) = 0;

        virtual ISceneNode* VFindActorNode(ActorId actorid) = 0;

        virtual const Frustum* VGetFrustum(void) = 0;

        virtual void VPushFrustum(Frustum* f) = 0;

        virtual void VSetParent(ISceneNode* parent, ISceneNode* child) = 0;

        virtual void VReleaseParent(ISceneNode* parent, ISceneNode* child) = 0;

        virtual void VPopFrustum(void) = 0;

        virtual void VResetVisibility(void) = 0;

        virtual bool VIsVisibilityReset(void) = 0;

        virtual ~ISceneGraph(void) {}
    };

    class ISceneNode
    {
    public:
        virtual util::Mat4* VGetTransformation(void) = 0;

        virtual void VSetVisibilityOnLastTraverse(bool visible) = 0;

        virtual void VSetActor(ActorId id) = 0;

        virtual ISceneNode* VGetParent(void) = 0;

        virtual bool VWasVisibleOnLastTraverse(void) = 0;

        virtual void VForceVisibilityCheck(void) = 0;

        virtual void VPreRender(ISceneGraph* graph) = 0;

        virtual void VSetParent(ISceneNode* parent) = 0;

        virtual void VRender(ISceneGraph* graph, RenderPath& path) = 0;

        virtual void _VRender(ISceneGraph* graph, RenderPath& path) = 0;

        virtual void VPostRender(ISceneGraph* graph) = 0;

        virtual void VOnRestore(ISceneGraph* graph) = 0;

        virtual void VAddChild(std::unique_ptr<ISceneNode> child) = 0;

        virtual void VOnParentChanged(void) = 0;

        virtual bool VIsVisible(ISceneGraph* graph) = 0;

        virtual std::unique_ptr<ISceneNode> VRemoveChild(ISceneNode* child) = 0;

        virtual std::unique_ptr<ISceneNode> VRemoveChild(ActorId actorId) = 0;

        virtual void VSetRenderPaths(RenderPath paths) = 0;

        virtual uint VGetRenderPaths(void) = 0;

        virtual ActorId VGetActorId(void) = 0;

        virtual std::vector<std::unique_ptr<ISceneNode>>& VGetChilds(void) = 0;

        virtual void VOnUpdate(ulong millis, ISceneGraph* graph) = 0;

        virtual const util::AxisAlignedBB& VGetAABB(void) const = 0;

        virtual void VQueryGeometry(IGeometry** geo) = 0;

        virtual ~ISceneNode(void) {}
    };

    class ICamera 
    {
    public:
        virtual const util::Mat4& GetView(void) = 0;

        virtual const util::Mat4& GetIView(void) = 0;

        virtual const util::Mat4& GetProjection(void) = 0;

        virtual const util::Mat4& GetViewProjection(void) = 0;

        virtual const util::Vec3& GetEyePos(void) = 0;

        virtual const util::Vec3& GetViewDir(void) = 0;

        virtual const util::Vec3& GetViewDirXZ(void) = 0;

        virtual const util::Vec3& GetSideDir(void) = 0;

        virtual const util::Vec3& GetUpDir(void) = 0;

        virtual float GetPhi(void) const = 0;

        virtual float GetTheta(void) const = 0;

        virtual void SetProjectionType(ProjectionType type) = 0;

        virtual void LookAt(const util::Vec3& eyePos, const util::Vec3& at) = 0;

        virtual void SetEyePos(const util::Vec3& pos) = 0;

        virtual void Move(float dx, float dy, float dz) {}

        virtual void Move(const util::Vec3& dt) {}

        virtual void Rotate(float dPhi, float dTheta) {}

        virtual void SetAspect(uint width, uint height) {}

        virtual float GetAspect(void) { return 0; }

        virtual void SetFoV(float foV) {}

        virtual float GetFoV(void) { return 0; }

        virtual void SetFar(float f) {}

        virtual float GetNear(void) { return 0; }

        virtual void SetNear(float n){}

        virtual float GetFar(void) { return 0; }

        virtual void MoveToPosition(const util::Vec3& pos) {}

        virtual void SetRotation(float phi, float theta) {}

        virtual void FromViewUp(const util::Vec3& up, const util::Vec3& dir) {}

        virtual void SetYOffset(float offset) { }

        virtual float GetYOffset(void) const { return 0; }

        virtual void SetPerspectiveProjection(float aspect, float fov, float fnear, float ffar) {}

        virtual void SetOrthographicProjection(float width, float height, float fnear, float ffar) {}

        virtual void SetPrthographicProjectionOffCenter(float left, float right, float up, float down, float fNear, float fFar) {}

        virtual Frustum& GetFrustum(void) = 0;

        virtual ~ICamera(void) {}
    };

    class IActorController : public IView, public IKeyListener, public IMouseListener 
    {
    public:
        virtual void VSetMinSpeed(float minSpeed) = 0;

        virtual void VSetMaxSpeed(float maxSpeed) = 0;

        virtual void VRegisterKeyPressedCommand(uint key, const std::string& command) = 0;

        virtual void VRegisterKeyReleasedCommand(uint key, const std::string& command) = 0;
        
        virtual void VRegisterKeyDownCommand(uint key, const std::string& command) = 0;

        virtual void VRegisterKeyCommand(uint key, const std::string& command) = 0;

        virtual void VSetMouseScrollAction(MouseScrollAction action) = 0;

        virtual void VSetUpdateAction(UpdateAction action) = 0;

        virtual void VActivate(void) = 0;

        virtual void VDeactivate(void) = 0;

        virtual ~IActorController(void) {}
    };

    class IHumanViewFactory
    {
    public:
         virtual IHumanView* VCreateHumanView(void) = 0;
    };

    class ISceneFactory
    {
    public:
        virtual ISceneNode* VCreateSceneNode(void) = 0;
        
        virtual ISceneGraph* VCreateSceneGraph(void) = 0;
    };

    typedef void (*EffectDrawMethod) (void);
    class IEffectParmaters
    {
    public:
        virtual void VApply(void) = 0;
    };

    class IEffect
    {
    public:
        virtual void VSetParameters(IEffectParmaters* params) = 0;

        virtual void VCreate(const CMShaderDescription& shaderDesc, float w, float h) = 0;

        virtual void VSetDrawMethod(EffectDrawMethod m) = 0;

        virtual void VAddRequirement(IEffect* effect) = 0;

        virtual void VSetSource(IRenderTarget* src) = 0;

        virtual void VProcess(void) = 0;

        virtual void VReset(void) = 0;

        virtual IRenderTarget* VGetSource(void) = 0;

        virtual float2 VGetViewPort(void) = 0;

        virtual void VSetTarget(IRenderTarget* target) = 0;

        virtual void VSetTarget(std::unique_ptr<IRenderTarget> target) = 0;

        virtual IRenderTarget* VGetTarget(void) = 0;

        virtual bool VOnRestore(uint w, uint h, ErrorLog* log = NULL) = 0;

        virtual ~IEffect(void) {}
    };

    class IEffectChain
    {
    public:
        virtual void VOnRestore(uint w, uint h) = 0;

        virtual void VSetSource(IRenderTarget* src) = 0;

        virtual void VSetTarget(IRenderTarget* target) = 0;

        virtual IEffect* VAppendEffect(const CMShaderDescription& shaderDesc, float percentofw = 1.0f, float percentofh = 1.0f) = 0;

        virtual IRenderTarget* VGetResult(void) = 0;

        virtual void VProcess(void) = 0;

        virtual ~IEffectChain(void) {}
    };

    class IEffectFactory
    {
    public:
        virtual IEffectChain* VCreateEffectChain(void) = 0;

        virtual IEffect* VCreateEffect(void) = 0;

        virtual ~IEffectFactory(void) {}
    };

    class IEffectFactoryFactory
    {
    public:
        virtual IEffectFactory* VCreateEffectFactory(void) = 0;
    };

    class IGraphicSetting
    {
    protected:
        std::string m_name;
    public:
        IGraphicSetting(LPCSTR settingName) : m_name(settingName) {}

        virtual void VRender(void) = 0;

        virtual bool VOnRestore(uint w, uint h) = 0;

        LPCSTR GetName(void) { return m_name.c_str(); }

        virtual CMShaderProgramDescription* VGetProgramDescription(void) = 0;

        virtual ~IGraphicSetting(void) {}
    };

    class IPostFXSetting : public IGraphicSetting
    {
    public:
        IPostFXSetting(LPCSTR name) : IGraphicSetting(name)
        {

        }

        virtual void VSetSource(IRenderTarget* src) = 0;

        virtual void VSetTarget(IRenderTarget* target) = 0;

        virtual IEffectChain* VGetEffectChain(void) = 0;

        virtual ~IPostFXSetting(void) {}
    };

    class IGraphicsSettings
    {
    public:
        virtual void VAddSetting(std::unique_ptr<IGraphicSetting> setting, GraphicsSettingType type) = 0;

        virtual void VSetPostFX(std::unique_ptr<IPostFXSetting> setting) = 0;

        virtual void VRender(void) = 0;

        virtual bool VOnRestore(uint w, uint h) = 0;

        virtual void VOnActivate(void) = 0;

        virtual IPostFXSetting* VGetPostFX(void) = 0;

        virtual IRenderTarget* VGetResult(void) = 0;

        virtual ~IGraphicsSettings(void) {}
    };

    class IEnvironmentLighting
    {
    public:

        virtual bool VOnRestore(void) = 0;

        virtual void VRender(ISceneGraph* graph) = 0;

        virtual UCHAR VGetSlices(void) = 0;

        virtual IRenderTarget** VGetTargets(void) = 0;

        virtual ~IEnvironmentLighting(void) {}
    };
}