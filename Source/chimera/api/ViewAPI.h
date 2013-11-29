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
        IView(VOID) : m_id((ViewId)-1), m_actor(NULL), m_name("undefined")
        {

        }

        IActor* VGetTarget(VOID)
        {
            return m_actor;
        }

        virtual VOID VOnAttach(UINT viewId, IActor* actor) 
        {
            m_id = viewId;
            VSetTarget(actor);
        }

        virtual VOID VSetTarget(IActor* actor)
        {
            if(actor)
            {
                m_actor = actor;
            }
        }

        ViewId GetId(VOID) CONST { return m_id; }

        CONST std::string& GetName(VOID) CONST { return m_name; }

        VOID SetName(CONST std::string& name) { m_name = name; }

        virtual BOOL VOnRestore(VOID) { return TRUE; };

        virtual BOOL VInitialise(FactoryPtr* facts) = 0;

        virtual VOID VOnUpdate(ULONG deltaMillis) = 0;

        virtual ViewType VGetType(VOID) CONST = 0;

        virtual ~IView(VOID) {}
    };

    class IHumanView : public IView
    {
    public:
        virtual IRenderer* VGetRenderer(VOID) = 0;

        virtual IPicker* VGetPicker(VOID) = 0;

        virtual IFontManager* VGetFontManager(VOID) = 0;

        virtual ISoundEngine* VGetSoundEngine(VOID) = 0;

        virtual ISoundSystem* VGetSoundSystem(VOID) = 0;

        virtual VOID VAddScreenElement(std::unique_ptr<IScreenElement> element) = 0;

        virtual IGuiFactory* VGetGuiFactory(VOID) = 0;

        virtual ISceneGraph* VGetSceneGraph(VOID) = 0;

        virtual VOID VOnRender(VOID) = 0;

        virtual IGraphicsFactory* VGetGraphicsFactory(VOID) = 0;

        virtual VOID VAddScene(std::unique_ptr<IRenderScreen> screen) = 0;

        virtual VOID VAddSceneNodeCreator(SceneNodeCreator nc, ComponentId cmpid) = 0;

        virtual VOID VRemoveSceneNodeCreator(ComponentId cmpid) = 0;

        virtual VOID VActivateScene(LPCSTR name) = 0;

        virtual IRenderScreen* VGetSceneByName(LPCSTR name) = 0;

        virtual IScreenElement* VGetScreenElementByName(LPCSTR name) = 0;

        virtual VOID VOnResize(UINT w, UINT h) = 0;

        virtual IEffectFactory* VGetEffectFactory(VOID) = 0;

        virtual IVRamManager* VGetVRamManager(VOID) = 0;
    };

    class ISceneGraph
    {
    public:
        virtual VOID VAddChild(ActorId actorid, std::unique_ptr<ISceneNode> node) = 0;

        virtual VOID VRemoveChild(ActorId actorid) = 0;

        virtual BOOL VOnRender(RenderPath path) = 0;

        virtual BOOL VOnRestore(VOID) = 0;

        virtual BOOL VOnUpdate(ULONG millis) = 0;

        virtual std::shared_ptr<ICamera> VGetCamera(VOID) = 0;

        virtual VOID VSetCamera(std::shared_ptr<ICamera> camera) = 0;

        virtual ISceneNode* VFindActorNode(ActorId actorid) = 0;

        virtual CONST Frustum* VGetFrustum(VOID) = 0;

        virtual VOID VPushFrustum(Frustum* f) = 0;

        virtual VOID VPopFrustum(VOID) = 0;

        virtual VOID VResetVisibility(VOID) = 0;

        virtual BOOL VIsVisibilityReset(VOID) = 0;

        virtual ~ISceneGraph(VOID) {}
    };

    class ISceneNode
    {
    public:
        virtual util::Mat4* VGetTransformation(VOID) = 0;

        virtual VOID VSetVisibilityOnLastTraverse(BOOL visible) = 0;

        virtual VOID VSetActor(ActorId id) = 0;

        virtual BOOL VWasVisibleOnLastTraverse(VOID) = 0;

        virtual VOID VForceVisibilityCheck(VOID) = 0;

        virtual VOID VPreRender(ISceneGraph* graph) = 0;

        virtual VOID VSetParent(ISceneNode* parent) = 0;

        virtual VOID VRender(ISceneGraph* graph, RenderPath& path) = 0;

        virtual VOID _VRender(ISceneGraph* graph, RenderPath& path) = 0;

        virtual VOID VRenderChildren(ISceneGraph* graph, RenderPath& path) = 0;

        virtual VOID VPostRender(ISceneGraph* graph) = 0;

        virtual VOID VOnRestore(ISceneGraph* graph) = 0;

        virtual VOID VAddChild(std::unique_ptr<ISceneNode> child) = 0;

        virtual VOID VOnParentChanged(VOID) = 0;

        virtual BOOL VIsVisible(ISceneGraph* graph) = 0;

        virtual BOOL VRemoveChild(ISceneNode* child) = 0;

        virtual BOOL VRemoveChild(ActorId actorId) = 0;

        virtual VOID VSetRenderPaths(RenderPath paths) = 0;

        virtual UINT VGetRenderPaths(VOID) = 0;

        virtual ActorId VGetActorId(VOID) = 0;

        virtual std::vector<std::unique_ptr<ISceneNode>>& VGetChilds(VOID) = 0;

        virtual VOID VOnUpdate(ULONG millis, ISceneGraph* graph) = 0;

        virtual ISceneNode* VFindActor(ActorId id) = 0;

        virtual CONST util::AxisAlignedBB& VGetAABB(VOID) CONST = 0;

        virtual VOID VQueryGeometry(IGeometry** geo) = 0;

        virtual ~ISceneNode(VOID) {}
    };

    class ICamera 
    {
    public:
        virtual CONST util::Mat4& GetView(VOID) = 0;

        virtual CONST util::Mat4& GetIView(VOID) = 0;

        virtual CONST util::Mat4& GetProjection(VOID) = 0;

        virtual CONST util::Mat4& GetViewProjection(VOID) = 0;

        virtual CONST util::Vec3& GetEyePos(VOID) = 0;

        virtual CONST util::Vec3& GetViewDir(VOID) = 0;

        virtual CONST util::Vec3& GetViewDirXZ(VOID) = 0;

        virtual CONST util::Vec3& GetSideDir(VOID) = 0;

        virtual CONST util::Vec3& GetUpDir(VOID) = 0;

        virtual FLOAT GetPhi(VOID) CONST = 0;

        virtual FLOAT GetTheta(VOID) CONST = 0;

        virtual VOID SetProjectionType(ProjectionType type) = 0;

        virtual VOID LookAt(CONST util::Vec3& eyePos, CONST util::Vec3& at) = 0;

        virtual VOID SetEyePos(CONST util::Vec3& pos) = 0;

        virtual VOID Move(FLOAT dx, FLOAT dy, FLOAT dz) {}

        virtual VOID Move(CONST util::Vec3& dt) {}

        virtual VOID Rotate(FLOAT dPhi, FLOAT dTheta) {}

        virtual VOID SetAspect(UINT width, UINT height) {}

        virtual FLOAT GetAspect(VOID) { return 0; }

        virtual VOID SetFoV(FLOAT foV) {}

        virtual FLOAT GetFoV(VOID) { return 0; }

        virtual VOID SetFar(FLOAT f) {}

        virtual FLOAT GetNear(VOID) { return 0; }

        virtual VOID SetNear(FLOAT n){}

        virtual FLOAT GetFar(VOID) { return 0; }

        virtual VOID MoveToPosition(CONST util::Vec3& pos) {}

        virtual VOID SetRotation(FLOAT phi, FLOAT theta) {}

        virtual VOID FromViewUp(CONST util::Vec3& up, CONST util::Vec3& dir) {}

        virtual VOID VSetYOffset(FLOAT offset) { }

        virtual VOID SetPerspectiveProjection(FLOAT aspect, FLOAT fov, FLOAT fnear, FLOAT ffar) {}

        virtual VOID SetOrthographicProjection(FLOAT width, FLOAT height, FLOAT fnear, FLOAT ffar) {}

        virtual VOID SetPrthographicProjectionOffCenter(FLOAT left, FLOAT right, FLOAT up, FLOAT down, FLOAT fNear, FLOAT fFar) {}

        virtual Frustum& GetFrustum(VOID) = 0;

        virtual ~ICamera(VOID) {}
    };

    class IParticleModifier
    {
    public:
        virtual VOID VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt) = 0;

        virtual VOID VOnRestore(ParticleSystem* sys)  = 0;

        virtual UINT VGetByteCount(VOID) = 0;

        virtual VOID VSetAABB(CONST util::AxisAlignedBB& aabb) = 0;
    };

    class IActorController : public IView, public IKeyListener, public IMouseListener 
    {
    public:
        virtual VOID VSetMinSpeed(FLOAT minSpeed) = 0;

        virtual VOID VSetMaxSpeed(FLOAT maxSpeed) = 0;

        virtual VOID VRegisterKeyPressedCommand(UINT key, CONST std::string& command) = 0;

        virtual VOID VRegisterKeyReleasedCommand(UINT key, CONST std::string& command) = 0;
        
        virtual VOID VRegisterKeyDownCommand(UINT key, CONST std::string& command) = 0;

        virtual VOID VRegisterKeyCommand(UINT key, CONST std::string& command) = 0;

        virtual VOID VSetMouseScrollAction(MouseScrollAction action) = 0;

        virtual VOID VSetUpdateAction(UpdateAction action) = 0;

        virtual VOID VActivate(VOID) = 0;

        virtual VOID VDeactivate(VOID) = 0;

        virtual ~IActorController(VOID) {}
    };

    class IHumanViewFactory
    {
    public:
         virtual IHumanView* VCreateHumanView(VOID) = 0;
    };

    class ISceneFactory
    {
    public:
        virtual ISceneNode* VCreateSceneNode(VOID) = 0;
        
        virtual ISceneGraph* VCreateSceneGraph(VOID) = 0;
    };

    typedef VOID (*EffectDrawMethod) (VOID);
    class IEffectParmaters
    {
    public:
        virtual VOID VApply(VOID) = 0;
    };

    class IEffect
    {
    public:
        virtual VOID VSetParameters(IEffectParmaters* params) = 0;

        virtual VOID VCreate(CONST CMShaderDescription& shaderDesc, FLOAT w, FLOAT h) = 0;

        virtual VOID VSetDrawMethod(EffectDrawMethod m) = 0;

        virtual VOID VAddRequirement(IEffect* effect) = 0;

        virtual VOID VSetSource(IRenderTarget* src) = 0;

        virtual VOID VProcess(VOID) = 0;

        virtual VOID VReset(VOID) = 0;

        virtual FLOAT2 VGetViewPort(VOID) = 0;

        virtual VOID VSetTarget(IRenderTarget* target) = 0;

        virtual VOID VSetTarget(std::unique_ptr<IRenderTarget> target) = 0;

        virtual IRenderTarget* VGetTarget(VOID) = 0;

        virtual BOOL VOnRestore(UINT w, UINT h, ErrorLog* log = NULL) = 0;

        virtual ~IEffect(VOID) {}
    };

    class IEffectChain
    {
    public:
        virtual VOID VOnRestore(UINT w, UINT h) = 0;

        virtual VOID VSetSource(IRenderTarget* src) = 0;

        virtual VOID VSetTarget(IRenderTarget* target) = 0;

        virtual IEffect* VCreateEffect(CONST CMShaderDescription& shaderDesc, FLOAT percentofw = 1.0f, FLOAT percentofh = 1.0f) = 0;

        virtual IRenderTarget* VGetResult(VOID) = 0;

        virtual VOID VProcess(VOID) = 0;

        virtual ~IEffectChain(VOID) {}
    };

    class IEffectFactory
    {
    public:
        virtual IEffectChain* VCreateEffectChain(VOID) = 0;

        virtual IEffect* VCreateEffect(VOID) = 0;

        virtual ~IEffectFactory(VOID) {}
    };

    class IEffectFactoryFactory
    {
    public:
        virtual IEffectFactory* VCreateEffectFactroy(VOID) = 0;
    };

    class IGraphicSetting
    {
    protected:
        std::string m_name;
    public:
        IGraphicSetting(LPCSTR settingName) : m_name(settingName) {}

        virtual VOID VRender(VOID) = 0;

        virtual BOOL VOnRestore(UINT w, UINT h) = 0;

        LPCSTR GetName(VOID) { return m_name.c_str(); }

        virtual CMShaderProgramDescription* VGetProgramDescription(VOID) = 0;

        virtual ~IGraphicSetting(VOID) {}
    };

    class IPostFXSetting : public IGraphicSetting
    {
    public:
        IPostFXSetting(LPCSTR name) : IGraphicSetting(name)
        {

        }

        virtual VOID VSetSource(IRenderTarget* src) = 0;

        virtual VOID VSetTarget(IRenderTarget* target) = 0;

        virtual ~IPostFXSetting(VOID) {}
    };

    class IGraphicsSettings
    {
    public:
        virtual VOID VAddSetting(std::unique_ptr<IGraphicSetting> setting, GraphicsSettingType type) = 0;

        virtual VOID VSetPostFX(std::unique_ptr<IPostFXSetting> setting) = 0;

        virtual VOID VRender(VOID) = 0;

        virtual BOOL VOnRestore(UINT w, UINT h) = 0;

        virtual VOID VOnActivate(VOID) = 0;

        virtual IRenderTarget* VGetResult(VOID) = 0;

        virtual ~IGraphicsSettings(VOID) {}
    };

    class IEnvironmentLighting
    {
    public:

        virtual BOOL VOnRestore(VOID) = 0;

        virtual VOID VRender(ISceneGraph* graph) = 0;

        virtual UCHAR VGetSlices(VOID) = 0;

        virtual IRenderTarget** VGetTargets(VOID) = 0;

        virtual ~IEnvironmentLighting(VOID) {}
    };
}