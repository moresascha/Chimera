#pragma once
#include "stdafx.h"
#include "Mat4.h"
#include "Actor.h"
#include "Vec3.h"
#include "Vec4.h"
#include "Resources.h"
#include "SceneNode.h"

namespace tinyxml2
{
    class XMLElement;
    class XMLDocument;
}

namespace util
{
    class ICamera;
}

namespace tbd 
{
    class ISceneNode;
    class Actor;

    class ActorComponent
    {
        friend class ActorFactory;
    private:
        BOOL m_waitTillHandled;
    protected:
        VOID SetOwner(std::weak_ptr<tbd::Actor> pOwner) { m_owner = pOwner; }
        std::weak_ptr<tbd::Actor> m_owner;
        HANDLE m_handle;

    public:
        ActorComponent(VOID);
        virtual BOOL VInit(tinyxml2::XMLElement* pData) { return TRUE; }
        virtual VOID VSave(tinyxml2::XMLDocument* pData) CONST {}
        virtual VOID VPostInit(VOID);
        virtual VOID VUpdate(ULONG millis) {}
        virtual VOID VDestroy(VOID) {}
        virtual VOID VCreateResources(VOID) {}
        virtual LPCSTR VGetName(VOID) = 0;
        BOOL IsWaitTillHandled(VOID) { return m_waitTillHandled; }
        HANDLE GetHandle(VOID) { return m_handle; }
        VOID WaitTillHandled(VOID) { m_waitTillHandled = TRUE; }
        VOID VSetHandled(VOID);
        virtual ComponentId GetComponentId(VOID) CONST = 0;
        virtual ~ActorComponent(VOID);
    };

    class TransformComponent : public ActorComponent
    {
    private:
        util::Mat4 m_transformation;
    public:
        CONST static ComponentId COMPONENT_ID;

        BOOL VInit(tinyxml2::XMLElement* pData);

        util::Mat4* GetTransformation(VOID) {
            return &m_transformation;
        }
        VOID VSave(tinyxml2::XMLDocument* pData) CONST;
        ComponentId GetComponentId(VOID) CONST { return COMPONENT_ID; }

        LPCSTR VGetName(VOID) { return "TransformComponent"; }
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

        std::string m_info;

        std::shared_ptr<tbd::ISceneNode> m_sceneNode;

        RenderComponent(VOID);

        tbd::Resource m_meshFile;

        CONST static ComponentId COMPONENT_ID;

        BOOL VInit(tinyxml2::XMLElement* pData);

        VOID VSave(tinyxml2::XMLDocument* pData) CONST;

        ComponentId GetComponentId(VOID) CONST { return COMPONENT_ID; }

        VOID VCreateResources(VOID);

        LPCSTR VGetName(VOID) { return "RenderComponent"; }
    };

    class CameraComponent : public ActorComponent
    {
    private:
        std::shared_ptr<util::ICamera> m_camera;
        std::string m_type;
    public:
        CONST static ComponentId COMPONENT_ID;

        BOOL VInit(tinyxml2::XMLElement* pData);

        std::shared_ptr<util::ICamera> GetCamera(VOID) { return m_camera; }

        ComponentId GetComponentId(VOID) CONST { return COMPONENT_ID; }

        LPCSTR VGetName(VOID) { return "CameraComponent"; }
    };

    class PhysicComponent : public ActorComponent 
    {

    public:
        std::string m_shapeType;
        std::string m_material;

        tbd::Resource m_meshFile;

        FLOAT m_radius;
        util::Vec3 m_dim;

        PhysicComponent(VOID) : m_dim(1,1,1), m_radius(1)
        {
            WaitTillHandled();
        }

        CONST static ComponentId COMPONENT_ID;

        BOOL VInit(tinyxml2::XMLElement* pData);

        VOID VSave(tinyxml2::XMLDocument* pData) CONST;

        ComponentId GetComponentId(VOID) CONST { return COMPONENT_ID; }

        VOID VCreateResources(VOID);

        LPCSTR VGetName(VOID) { return "PhysicComponent"; }
    };

    class LightComponent : public ActorComponent
    {

    public:
        std::string m_type;
        util::Vec4 m_color;
        FLOAT m_phi;
        FLOAT m_theta;
        BOOL m_activated;

        LightComponent(VOID)
        {
            WaitTillHandled();
        }

        CONST static ComponentId COMPONENT_ID;

        BOOL VInit(tinyxml2::XMLElement* pData);

        VOID VSave(tinyxml2::XMLDocument* pData) CONST;

        ComponentId GetComponentId(VOID) CONST { return COMPONENT_ID; }

        LPCSTR VGetName(VOID) { return "LightComponent"; }
    };

    class PickableComponent : public ActorComponent
    {
    public:
        CONST static ComponentId COMPONENT_ID;
        ComponentId GetComponentId(VOID) CONST { return COMPONENT_ID; }
        BOOL VInit(tinyxml2::XMLElement* pData) { return TRUE; }
        VOID VSave(tinyxml2::XMLDocument* pData) CONST;
        LPCSTR VGetName(VOID) { return "PickableComponent"; }
    };

    class ParticleComponent : public ActorComponent 
    {
    public:
        CONST static ComponentId COMPONENT_ID;
        ComponentId GetComponentId(VOID) CONST { return COMPONENT_ID; }
        BOOL VInit(tinyxml2::XMLElement* pData);
        VOID VSave(tinyxml2::XMLDocument* pData) CONST;
        LPCSTR VGetName(VOID) { return "ParticleComponent"; }
    };

    class SoundEmitterComponent : public ActorComponent
    {
    public:
        SoundEmitterComponent(VOID);
        std::string m_soundFile;
        FLOAT m_radius;
        BOOL m_loop;
        CONST static ComponentId COMPONENT_ID;
        ComponentId GetComponentId(VOID) CONST { return COMPONENT_ID; }
        BOOL VInit(tinyxml2::XMLElement* pData);
        VOID VSave(tinyxml2::XMLDocument* pData) CONST;
        VOID VCreateResources(VOID);
        LPCSTR VGetName(VOID) { return "SoundEmitterComponent"; }
    };
}