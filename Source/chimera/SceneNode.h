#pragma once
#include "stdafx.h"
#include "AxisAlignedBB.h"
#include "Cache.h"

namespace chimera 
{
    class TransformComponent;
    //helper for nodes
    /*
    VOID DrawPickingSphere(std::shared_ptr<chimera::Actor> actor, CONST util::Mat4* matrix, FLOAT radius);
    VOID DrawPickingCube(std::shared_ptr<chimera::Actor> actor, CONST util::Mat4* matrix, CONST util::AxisAlignedBB& aabb);
    VOID DrawActorInfos(std::shared_ptr<chimera::Actor> actor, CONST util::Mat4* matrix, std::shared_ptr<chimera::ICamera> camera);
    VOID DrawSphere(CONST util::Mat4* matrix, CONST util::AxisAlignedBB& aabb);
    VOID DrawSphere(CONST util::Mat4* matrix, CONST FLOAT radius);
    VOID DrawAnchorSphere(std::shared_ptr<chimera::Actor> actor, CONST util::Mat4* matrix, FLOAT radius);
    VOID DrawBox(CONST util::Mat4* matrix, CONST util::AxisAlignedBB& aabb);
    VOID DrawBox(CONST util::AxisAlignedBB& aabb);
    VOID DrawToShadowMap(std::shared_ptr<chimera::Geometry> geo, std::shared_ptr<chimera::Mesh> mesh, CONST util::Mat4* matrix);
    VOID DrawInfoTextOnScreen(chimera::ICamera* camera, CONST util::Mat4* model, CONST std::string& text);
    VOID DrawFrustum(chimera::Frustum& frustum); */

    void SetActorId(ActorId id);
    void SetActorId(const IActor* id);

    class SceneNode : public ISceneNode 
    {
    private:
        bool m_wasVisibleOnLastTraverse;
        bool m_forceVisibleCheck;
        TransformComponent* m_transformation;
        std::unique_ptr<TransformComponent> m_wParentTransformation;

    protected:
        ActorId m_actorId;
        IActor* m_actor;
        std::vector<std::unique_ptr<ISceneNode>> m_childs;
        util::AxisAlignedBB m_aabb;
        ISceneNode* m_parent;
        RenderPath m_paths;
        std::shared_ptr<IGeometry> m_pGeometry;

        bool HasParent(void);

    public:
        SceneNode(ActorId ActorId);

        SceneNode(void);

        void VQueryGeometry(IGeometry** geo);

        util::Mat4* VGetTransformation(void);

        ISceneNode* VGetParent(void);

        void VSetActor(ActorId id);

        void VSetParent(ISceneNode* parent);

        bool VWasVisibleOnLastTraverse(void);

        void VSetVisibilityOnLastTraverse(bool visible);

        void VForceVisibilityCheck(void);

        ActorId VGetActorId(void);

        void VOnParentChanged(void);

        std::vector<std::unique_ptr<ISceneNode>>& VGetChilds(void);

        virtual void VPreRender(ISceneGraph* graph);

        virtual void VPostRender(ISceneGraph* graph);

        virtual void VOnRestore(ISceneGraph* graph);

        void VRender(ISceneGraph* graph, RenderPath& path);

        virtual void _VRender(ISceneGraph* graph, RenderPath& path) {}

        virtual bool VIsVisible(ISceneGraph* graph);

        void VAddChild(std::unique_ptr<ISceneNode> child);

        std::unique_ptr<ISceneNode> VRemoveChild(ActorId actorId);

        std::unique_ptr<ISceneNode> VRemoveChild(ISceneNode* child);

        virtual const util::AxisAlignedBB& VGetAABB(void) const { return m_aabb; }

        virtual void VOnUpdate(ulong millis, ISceneGraph* graph);

        void ActorMovedDelegate(chimera::IEventPtr pEventData);

        virtual void VOnActorMoved(void);

        uint VGetRenderPaths(void);

        void VSetRenderPaths(RenderPath paths);

        virtual ~SceneNode(void);
    };

   
    //for mesh nodes only
    //VOID DrawPicking(std::shared_ptr<chimera::Actor> actor, CONST util::Mat4* matrix, std::shared_ptr<chimera::Mesh> mesh, std::shared_ptr<chimera::Geometry> geo);

    class MeshNode : public SceneNode 
    {
    protected:
        util::Vec3 m_transformedBBPoint;
        float m_longestScale;
        chimera::CMResource m_ressource;
        std::string m_meshId;
        std::shared_ptr<IMeshSet> m_meshSet;
        IMesh* m_mesh;
        std::shared_ptr<MaterialSet> m_materials;
        std::shared_ptr<IDeviceTexture> m_diffuseTextures[64];
        std::shared_ptr<IDeviceTexture> m_normalTextures[64];
        uint m_diffuseTexturesCount;
        void DrawToAlbedo(void);

    public:
        MeshNode(ActorId actorid, CMResource ressource, std::string meshId);

        virtual bool VIsVisible(ISceneGraph* graph);

        virtual void _VRender(ISceneGraph* graph, chimera::RenderPath& path);

        virtual void VOnActorMoved(void);

        virtual void VOnRestore(ISceneGraph* graph);

        virtual ~MeshNode(void);
    };

    class SkyDomeNode : public SceneNode
    {
    private:
        CMResource m_TextureRes;
        std::shared_ptr<IDeviceTexture> m_textureHandle;

    public:
        SkyDomeNode(ActorId id, CMResource texture);

        void _VRender(ISceneGraph* graph, RenderPath& path);

        bool VIsVisible(ISceneGraph* graph);

        void VOnRestore(ISceneGraph* graph);

        ~SkyDomeNode(void);
    };

    class InstancedMeshNode : public MeshNode
    {
    private:
        std::shared_ptr<IVertexBuffer> m_pInstances;

    public:
        InstancedMeshNode(ActorId actorid, std::shared_ptr<IVertexBuffer> instances, CMResource ressource);

        bool VIsVisible(ISceneGraph* graph);

        void _VRender(ISceneGraph* graph, RenderPath& path);

        void VOnActorMoved(void);

        void VOnRestore(ISceneGraph* graph);

        ~InstancedMeshNode(void);
    };

    /*
    enum AnchorMeshType
    {
        eBOX,
        eSPHERE
    };
    enum AnchroDrawMode
    {
        eFillMode_Solid,
        eFillMode_Wire
    };

    class AnchorNode : public SceneNode
    {
    private:
        AnchorMeshType m_meshType;
        FLOAT m_radius; //Todo
        AnchroDrawMode m_drawMode;
        std::string m_info;
    public:
        AnchorNode(AnchorMeshType meshType, ActorId id, LPCSTR info, FLOAT radius = 1.0f, AnchroDrawMode mode = eFillMode_Solid) 
            : SceneNode(id), m_meshType(meshType), m_radius(radius), m_drawMode(mode), m_info(info)
        {
        }

        VOID _VRender(chimera::SceneGraph* graph, chimera::RenderPath& path);

        UINT VGetRenderPaths(VOID);
    };
    */
    class CameraNode : public SceneNode
    {
    private:
        std::shared_ptr<chimera::ICamera> m_pCamera;
    public:
        CameraNode(ActorId id);

        void _VRender(ISceneGraph* graph, RenderPath& path);
    };
    
    class GeometryNode : public SceneNode
    {
    private:
        std::unique_ptr<IMaterial> m_pMaterial;

    public:
        GeometryNode(ActorId id, std::unique_ptr<IGeometry> geo);

        void SetMaterial(const IMaterial& mat);

        void _VRender(ISceneGraph* graph, RenderPath& path);

        void VOnRestore(ISceneGraph* graph);

        ~GeometryNode(void);
    };
}
