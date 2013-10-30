#pragma once
#include "stdafx.h"
#include "Mat4.h"

#define CM_CMP_TRANSFORM 0xdb756713
#define CM_CMP_RENDERING 0x8beb1acc
#define CM_CMP_CAMERA 0xb8a716ca
#define CM_CMP_PHX 0xc1514f
#define CM_CMP_LIGHT 0x1b5b0ea4
#define CM_CMP_PICKABLE 0xd295188c
#define CM_CMP_SOUND 0x568a0c05
#define CM_CMP_PARENT_ACTOR 0xde7b06f1

#define CM_CREATE_CMP_HEADER(__type, __name) \
    ComponentId VGetComponentId(VOID) CONST { return __type; } \
    LPCSTR VGetName(VOID) CONST { return #__name; }

namespace chimera 
{
    class ActorComponent : public IActorComponent
    {
    protected:
        IActor* m_owner;
    public:
        ActorComponent(VOID);

        VOID VSetOwner(IActor* pOwner) { m_owner = pOwner; }

        virtual BOOL VInitialize(IStream* stream) { return TRUE; }

        virtual VOID VSerialize(IStream* stream) CONST { }

        virtual VOID VPostInit(VOID);

        virtual VOID VCreateResources(VOID) { }

        IActor* VGetActor(VOID) { return m_owner; }

        virtual ~ActorComponent(VOID);
    };

    class TransformComponent : public ActorComponent
    {
    private:
        util::Mat4 m_transformation;
    public:
        FLOAT m_phi;
        FLOAT m_theta;

        TransformComponent(VOID);

        BOOL VInitialize(IStream* stream);

        VOID VSerialize(IStream* stream) CONST;

        util::Mat4* GetTransformation(VOID) {
            return &m_transformation;
        }

        CM_CREATE_CMP_HEADER(CM_CMP_TRANSFORM, TransformComponent);
    };

    class RenderComponent : public ActorComponent 
    {
    public:
        std::list<util::Vec3> m_instances;
        std::string m_anchorType;

        std::string m_type;

        std::string m_drawType;

        FLOAT m_anchorRadius;

        util::Vec3 m_anchorBoxHE;

        std::shared_ptr<chimera::ISceneNode> m_sceneNode;

        RenderComponent(VOID);

        CMResource m_resource;

        BOOL VInitialize(IStream* stream);

        VOID VSerialize(IStream* stream) CONST;

        VOID VCreateResources(VOID);

        CM_CREATE_CMP_HEADER(CM_CMP_RENDERING, RenderComponent);
    };

    class CameraComponent : public ActorComponent
    {
    private:
        std::shared_ptr<ICamera> m_camera;
        std::string m_type;
    public:
        CameraComponent(VOID)
        {

        }
        BOOL VInitialize(IStream* stream);

        std::shared_ptr<ICamera> GetCamera(VOID) { return m_camera; }

        VOID SetCamera(std::shared_ptr<ICamera> cam) { m_camera = cam; }

        CM_CREATE_CMP_HEADER(CM_CMP_CAMERA, CameraComponent);
    };

    class PhysicComponent : public ActorComponent 
    {

    public:
        std::string m_shapeStyle;
        std::string m_shapeType;
        std::string m_material;

        chimera::CMResource m_meshFile;

        FLOAT m_radius;
        util::Vec3 m_dim;

        PhysicComponent(VOID) : m_dim(1,1,1), m_radius(1)
        {

        }

        BOOL VInitialize(IStream* stream);

        VOID VSerialize(IStream* stream) CONST;

        VOID VCreateResources(VOID);

        CM_CREATE_CMP_HEADER(CM_CMP_PHX, PhysicComponent); 
    };

    class LightComponent : public ActorComponent
    {

    public:
        std::string m_type;
        util::Vec4 m_color;
        FLOAT m_angle;
        FLOAT m_intensity;
        BOOL m_activated;

        LightComponent(VOID) : m_angle(0), m_activated(TRUE), m_intensity(1)
        {

        }

        BOOL VInitialize(IStream* stream);

        VOID VSerialize(IStream* stream) CONST;

        CM_CREATE_CMP_HEADER(CM_CMP_LIGHT, LightComponent); 
    };

    class PickableComponent : public ActorComponent
    {
    public:
        BOOL VInit(tinyxml2::XMLElement* pData) { return TRUE; }
        VOID VSerialize(IStream* stream) CONST;

        CM_CREATE_CMP_HEADER(CM_CMP_PICKABLE, PickableComponent); 
    };

    class SoundComponent : public ActorComponent
    {
    public:
        SoundComponent(VOID);
        std::string m_soundFile;
        FLOAT m_radius;
        BOOL m_emitter;
        BOOL m_loop;
        BOOL VInitialize(IStream* stream);
        VOID VSerialize(IStream* stream) CONST;
        VOID VCreateResources(VOID);
        
        CM_CREATE_CMP_HEADER(CM_CMP_SOUND, SoundComponent); 
    };

    class ParentComponent : public ActorComponent
    {
    public:
        ActorId m_parentId;
    public:
        ActorId GetParent(VOID) { return m_parentId; }

        VOID VPostInit(VOID);

        CM_CREATE_CMP_HEADER(CM_CMP_PARENT_ACTOR, ParentComponent); 
    };
}