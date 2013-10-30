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

    class SceneNode : public ISceneNode 
    {
    private:
        BOOL m_wasVisibleOnLastTraverse;
        BOOL m_forceVisibleCheck;
        TransformComponent* m_transformation;
        std::unique_ptr<TransformComponent> m_wParentTransformation;

    protected:
        ActorId m_actorId;
        IActor* m_actor;
        std::vector<std::unique_ptr<ISceneNode>> m_childs;
        util::AxisAlignedBB m_aabb;
        ISceneNode* m_parent;
        RenderPath m_paths;

        BOOL HasParent(VOID);

    public:
        SceneNode(ActorId ActorId);

        SceneNode(VOID);

        util::Mat4* VGetTransformation(VOID);

        VOID VSetActor(ActorId id);

        VOID VSetParent(ISceneNode* parent);

        BOOL VWasVisibleOnLastTraverse(VOID);

        VOID VSetVisibilityOnLastTraverse(BOOL visible);

        VOID VForceVisibilityCheck(VOID);

        ActorId VGetActorId(VOID);

        VOID VOnParentChanged(VOID);

        std::vector<std::unique_ptr<ISceneNode>>& VGetChilds(VOID);

        virtual VOID VPreRender(ISceneGraph* graph);

        virtual VOID VPostRender(ISceneGraph* graph);

        virtual VOID VOnRestore(ISceneGraph* graph);

        VOID VRender(ISceneGraph* graph, RenderPath& path);

        virtual VOID _VRender(ISceneGraph* graph, RenderPath& path) {}

        VOID VRenderChildren(ISceneGraph* graph, RenderPath& path);

        virtual BOOL VIsVisible(ISceneGraph* graph);

        VOID VAddChild(std::unique_ptr<ISceneNode> child);

        BOOL VRemoveChild(ActorId actorId);

        BOOL VRemoveChild(ISceneNode* child);

        virtual CONST util::AxisAlignedBB& VGetAABB(VOID) CONST { return m_aabb; }

        virtual VOID VOnUpdate(ULONG millis, ISceneGraph* graph);

        VOID ActorMovedDelegate(chimera::IEventPtr pEventData);

        virtual VOID VOnActorMoved(VOID) {}

        UINT VGetRenderPaths(VOID);

        VOID VSetRenderPaths(RenderPath paths);

        ISceneNode* VFindActor(ActorId id);

        virtual ~SceneNode(VOID);
    };

   
    //for mesh nodes only
    //VOID DrawPicking(std::shared_ptr<chimera::Actor> actor, CONST util::Mat4* matrix, std::shared_ptr<chimera::Mesh> mesh, std::shared_ptr<chimera::Geometry> geo);

    class MeshNode : public SceneNode 
    {
    protected:
        util::Vec3 m_transformedBBPoint;
        chimera::CMResource m_ressource;
        std::shared_ptr<IGeometry> m_geo;
        std::shared_ptr<IMesh> m_mesh;
        std::shared_ptr<MaterialSet> m_materials;
        std::shared_ptr<IDeviceTexture> m_diffuseTextures[32];
        std::shared_ptr<IDeviceTexture> m_normalTextures[32];
        UINT m_diffuseTexturesCount;
        VOID DrawToAlbedo(VOID);

    public:
        MeshNode(ActorId actorid, chimera::CMResource ressource);

        virtual BOOL VIsVisible(ISceneGraph* graph);

        virtual VOID _VRender(ISceneGraph* graph, chimera::RenderPath& path);

        virtual VOID VOnActorMoved(VOID);

        virtual VOID VOnRestore(ISceneGraph* graph);

        virtual ~MeshNode(VOID);
    };

    class SkyDomeNode : public SceneNode
    {
    private:
        CMResource m_TextureRes;
        std::shared_ptr<IDeviceTexture> m_textureHandle;
        IGeometry* m_pSkyGeo;
    public:

        SkyDomeNode(ActorId id, CMResource texture);

        VOID _VRender(ISceneGraph* graph, RenderPath& path);

        BOOL VIsVisible(ISceneGraph* graph);

        VOID VOnRestore(ISceneGraph* graph);

        ~SkyDomeNode(VOID);
    };

    /*
    class InstancedMeshNode : public MeshNode
    {
    private:
        std::shared_ptr<chimera::VertexBufferHandle> m_pInstanceHandle;

    public:
        InstancedMeshNode(ActorId actorid, chimera::CMResource ressource);

        BOOL VIsVisible(SceneGraph* graph);

        VOID _VRender(chimera::SceneGraph* graph, chimera::RenderPath& path);

        VOID VOnActorMoved(VOID);

        VOID VOnRestore(chimera::SceneGraph* graph);

        UINT VGetRenderPaths(VOID);

        ~InstancedMeshNode(VOID);
    };

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

    class CameraNode : public SceneNode
    {
    private:
        std::shared_ptr<chimera::ICamera> m_pCamera;

    public:
        CameraNode(ActorId id);

        VOID _VRender(chimera::SceneGraph* graph, chimera::RenderPath& path);

        UINT VGetRenderPaths(VOID);
    };

    typedef chimera::Geometry*(*GeoCreator)(VOID);

    class GeometryNode : public SceneNode
    {
        GeoCreator m_pFuncGeometry;
        std::shared_ptr<chimera::Geometry> m_pGeometry;
        chimera::Material* m_pMaterial;

    public:
        GeometryNode(GeoCreator gc);

        VOID SetMaterial(CONST chimera::Material& mat);

        VOID SetAABB(CONST util::AxisAlignedBB& aabb);

        VOID _VRender(chimera::SceneGraph* graph, chimera::RenderPath& path);

        VOID VOnRestore(chimera::SceneGraph* graph);

        UINT VGetRenderPaths(VOID);

        ~GeometryNode(VOID);
    }; */
}
