#pragma once
#include "stdafx.h"
#include "GameView.h"
#include "AxisAlignedBB.h"
#include "Actor.h"
#include "Resources.h"
#include "RenderPath.h"

namespace d3d
{
    class Geometry;
    class VertexBufferHandle;
    class Texture2D;
}

namespace tbd
{
    class Mesh;
    class Frustum;
    class MaterialSet;
    class TransformComponent;
    class Material;
}

namespace util
{
    class ICamera;
}

namespace tbd 
{
    class SceneGraph;
    class SceneNode;

    //helper for nodes
    VOID DrawPickingSphere(std::shared_ptr<tbd::Actor> actor, CONST util::Mat4* matrix, FLOAT radius);
    VOID DrawPickingCube(std::shared_ptr<tbd::Actor> actor, CONST util::Mat4* matrix, CONST util::AxisAlignedBB& aabb);
    VOID DrawActorInfos(std::shared_ptr<tbd::Actor> actor, CONST util::Mat4* matrix, std::shared_ptr<util::ICamera> camera);
    VOID DrawSphere(CONST util::Mat4* matrix, CONST util::AxisAlignedBB& aabb);
    VOID DrawSphere(CONST util::Mat4* matrix, CONST FLOAT radius);
    VOID DrawAnchorSphere(std::shared_ptr<tbd::Actor> actor, CONST util::Mat4* matrix, FLOAT radius);
    VOID DrawBox(CONST util::Mat4* matrix, CONST util::AxisAlignedBB& aabb);
    VOID DrawToShadowMap(std::shared_ptr<d3d::Geometry> geo, std::shared_ptr<tbd::Mesh> mesh, CONST util::Mat4* matrix);
    VOID DrawInfoTextOnScreen(util::ICamera* camera, CONST util::Mat4* model, CONST std::string& text);
    VOID DrawFrustum(tbd::Frustum& frustum);

    class ISceneNode
    {
    public:
        ISceneNode(VOID) {}

        virtual util::Mat4* GetTransformation(VOID) = 0;

        virtual VOID VSetVisibilityOnLastTraverse(BOOL visible) = 0;

        virtual VOID VSetActor(ActorId id) = 0;

        virtual BOOL VWasVisibleOnLastTraverse(VOID) = 0;

        virtual VOID VForceVisibilityCheck(VOID) = 0;

        virtual VOID VPreRender(SceneGraph* graph) = 0;

        virtual VOID VSetParent(ISceneNode* parent) = 0;

        virtual VOID VRender(SceneGraph* graph, tbd::RenderPath& path) = 0;

        virtual VOID _VRender(SceneGraph* graph, tbd::RenderPath& path) = 0;

        virtual VOID VRenderChildren(SceneGraph* graph, tbd::RenderPath& path) = 0;

        virtual VOID VPostRender(SceneGraph* graph) = 0;

        virtual VOID VOnRestore(SceneGraph* graph) = 0;

        virtual VOID VAddChild(std::shared_ptr<ISceneNode> child) = 0;

        virtual VOID VOnParentChanged(VOID) = 0;

        virtual BOOL VIsVisible(SceneGraph* graph) = 0;

        virtual BOOL VRemoveChild(std::shared_ptr<ISceneNode> child) = 0;

        virtual BOOL VRemoveChild(ActorId actorId) = 0;

        virtual UINT VGetRenderPaths(VOID) = 0;

        virtual ActorId VGetActorId(VOID) = 0;

        virtual std::vector<std::shared_ptr<ISceneNode>>& GetChilds(VOID) = 0;

        virtual VOID VOnUpdate(ULONG millis, SceneGraph* graph) = 0;

        virtual std::shared_ptr<tbd::ISceneNode> VFindActor(ActorId id) = 0;

        virtual ~ISceneNode(VOID) {}
    };

    class SceneNode : public ISceneNode 
    {
    private:
        BOOL m_wasVisibleOnLastTraverse;
        BOOL m_forceVisibleCheck;
        std::shared_ptr<tbd::TransformComponent> m_transformation;
        std::shared_ptr<tbd::TransformComponent> m_wParentTransformation;
    protected:
        ActorId m_actorId;
        std::shared_ptr<tbd::Actor> m_actor;
        std::vector<std::shared_ptr<ISceneNode>> m_childs;
        util::AxisAlignedBB m_aabb;
        ISceneNode* m_parent;

        BOOL HasParent(VOID);

    public:
        SceneNode(ActorId ActorId);

        SceneNode(VOID);

        util::Mat4* GetTransformation(VOID);

        VOID VSetActor(ActorId id);

        VOID VSetParent(ISceneNode* parent);

        BOOL VWasVisibleOnLastTraverse(VOID);

        VOID VSetVisibilityOnLastTraverse(BOOL visible);

        VOID VForceVisibilityCheck(VOID);

        ActorId VGetActorId(VOID);

        VOID VOnParentChanged(VOID);

        std::vector<std::shared_ptr<ISceneNode>>& GetChilds(VOID);

        virtual VOID VPreRender(tbd::SceneGraph* graph);

        virtual VOID VPostRender(tbd::SceneGraph* graph);

        virtual VOID VOnRestore(tbd::SceneGraph* graph);

        VOID VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);

        virtual VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path) {}

        VOID VRenderChildren(SceneGraph* graph, tbd::RenderPath& path);

        virtual BOOL VIsVisible(SceneGraph* graph);

        VOID VAddChild(std::shared_ptr<ISceneNode> child);

        BOOL VRemoveChild(ActorId actorId);

        BOOL VRemoveChild(std::shared_ptr<ISceneNode> child);

        virtual CONST util::AxisAlignedBB& GetAABB(VOID) CONST { return m_aabb; }

        virtual VOID VOnUpdate(ULONG millis, SceneGraph* graph);

        VOID ActorMovedDelegate(event::IEventPtr pEventData);

        virtual VOID VOnActorMoved(VOID) {}

        virtual UINT VGetRenderPaths(VOID);

        std::shared_ptr<tbd::ISceneNode> VFindActor(ActorId id);

        virtual ~SceneNode(VOID);
    };

    //for mesh nodes only
    VOID DrawPicking(std::shared_ptr<tbd::Actor> actor, CONST util::Mat4* matrix, std::shared_ptr<tbd::Mesh> mesh, std::shared_ptr<d3d::Geometry> geo);

    class MeshNode : public SceneNode 
    {
    protected:
        util::Vec3 m_transformedBBPoint;
        tbd::Resource m_ressource;
        std::shared_ptr<d3d::Geometry> m_geo;
        std::shared_ptr<tbd::Mesh> m_mesh;
        std::shared_ptr<tbd::MaterialSet> m_materials;
        std::shared_ptr<d3d::Texture2D> m_diffuseTextures[32];
        std::shared_ptr<d3d::Texture2D> m_normalTextures[32];
        UINT m_diffuseTexturesCount;
        VOID DrawToAlbedo(VOID);

    public:
        MeshNode(ActorId actorid, tbd::Resource ressource);

        MeshNode(tbd::Resource ressource);

        virtual BOOL VIsVisible(SceneGraph* graph);

        virtual VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);

        virtual VOID VOnActorMoved(VOID);

        virtual VOID VOnRestore(tbd::SceneGraph* graph);

        virtual UINT VGetRenderPaths(VOID);

        virtual ~MeshNode(VOID);
    };

    class InstancedMeshNode : public MeshNode
    {
    private:
        std::shared_ptr<d3d::VertexBufferHandle> m_pInstanceHandle;

    public:
        InstancedMeshNode(ActorId actorid, tbd::Resource ressource);

        BOOL VIsVisible(SceneGraph* graph);

        VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);

        VOID VOnActorMoved(VOID);

        VOID VOnRestore(tbd::SceneGraph* graph);

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
        eSolid,
        eWire
    };

    class AnchorNode : public SceneNode
    {
    private:
        AnchorMeshType m_meshType;
        FLOAT m_radius; //Todo
        AnchroDrawMode m_drawMode;
        std::string m_info;
    public:
        AnchorNode(AnchorMeshType meshType, ActorId id, LPCSTR info, FLOAT radius = 1.0f, AnchroDrawMode mode = eSolid) 
            : SceneNode(id), m_meshType(meshType), m_radius(radius), m_drawMode(mode), m_info(info)
        {
        }

        VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);

        UINT VGetRenderPaths(VOID);
    };

    class CameraNode : public SceneNode
    {
    private:
        std::shared_ptr<util::ICamera> m_pCamera;

    public:
        CameraNode(ActorId id);

        VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);

        UINT VGetRenderPaths(VOID);
    };

    class SkyDomeNode : public SceneNode
    {
    private:
        tbd::Resource m_TextureRes;
        std::shared_ptr<d3d::Texture2D> m_textureHandle;
    public:

        SkyDomeNode(ActorId id, tbd::Resource texture);

        VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);

        BOOL VIsVisible(SceneGraph* graph);

        VOID VOnRestore(tbd::SceneGraph* graph);

        UINT VGetRenderPaths(VOID);
    };

    class GeometryNode : public SceneNode
    {
        d3d::Geometry* m_pGeometry;
        tbd::Material* m_pMaterial;

    public:
        GeometryNode(d3d::Geometry* m_pGeometry);

        VOID SetMaterial(CONST tbd::Material* mat);

        VOID SetAABB(CONST util::AxisAlignedBB& aabb);

        VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path);

        UINT VGetRenderPaths(VOID);

        ~GeometryNode(VOID);
    };
}
