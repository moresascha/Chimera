#pragma once
#include "stdafx.h"
#include "Actor.h"
#include "ActorFactory.h"
#include "Event.h"

namespace tbd
{
    class ILevel
    {
    public:
        virtual BOOL VLoad(BOOL block) = 0;

        virtual VOID VUnload(VOID) = 0;

        virtual BOOL VSave(LPCSTR file = NULL) = 0;

        virtual UINT VGetActorsCount(VOID) = 0;

        virtual FLOAT VGetLoadingProgress(VOID) = 0;

        virtual std::shared_ptr<tbd::Actor> VAddActor(ActorDescription& desc) = 0;

        virtual VOID VRemoveActor(ActorId id) = 0;

        virtual std::shared_ptr<tbd::Actor> VFindActor(ActorId id) = 0;

        virtual ~ILevel(VOID) {}
    };

    class BaseLevel : public ILevel
    {
    private:
        std::string m_name;
        std::string m_file;
        
    protected:
        std::map<ActorId, std::shared_ptr<tbd::Actor>> m_actors;
        std::vector<ActorId> m_idsToLoad;
        tbd::ActorFactory* m_pActorFactory;

    public:

        BaseLevel::BaseLevel(CONST std::string& file, tbd::ActorFactory* factory);

        CONST std::string& GetName(VOID)
        {
            return m_name;
        }

        CONST std::string& GetFile(VOID)
        {
            return m_file;
        }

        VOID SetFile(CONST std::string& file)
        {
            m_file = file;
        }

        VOID SetName(CONST std::string& name)
        {
            m_name = name;
        }

        UINT VGetActorsCount(VOID);

        std::shared_ptr<tbd::Actor> VFindActor(ActorId id);

        VOID VRemoveActor(ActorId id);

        VOID ActorCreatedDelegate(event::IEventPtr eventData);

        FLOAT VGetLoadingProgress(VOID);

        VOID VUnload(VOID);

        std::shared_ptr<tbd::Actor> VAddActor(ActorDescription& desc);

        virtual ~BaseLevel(VOID);
    };

    class XMLLevel : public BaseLevel
    {
    private:

    public:
        XMLLevel(CONST std::string& file, tbd::ActorFactory* factory);
        
        BOOL VLoad(BOOL block);
        
        BOOL VSave(LPCSTR file = NULL);
                
        std::shared_ptr<tbd::Actor> VAddActor(tinyxml2::XMLElement* pNode);
    };

    class RandomLevel : public BaseLevel
    {
    public:
        RandomLevel(CONST std::string& file, tbd::ActorFactory* factory);

        BOOL VLoad(BOOL block);

        BOOL VSave(LPCSTR file = NULL);
    };

    class TransformShowRoom : public BaseLevel
    {
    public:
        TransformShowRoom(CONST std::string& file, tbd::ActorFactory* factory);

        BOOL VLoad(BOOL block);

        BOOL VSave(LPCSTR file = NULL);
    };

    class CudaTransformationNode;

    class BSplinePatchLevel : public BaseLevel
    {
    private:
        std::shared_ptr<tbd::Actor>* m_controlPoints;
        tbd::CudaTransformationNode* m_node;

    public:
        BSplinePatchLevel(CONST std::string& file, tbd::ActorFactory* factory);

        BOOL VLoad(BOOL block);

        BOOL VSave(LPCSTR file = NULL);

        VOID OnControlPointMove(event::IEventPtr data);

        ~BSplinePatchLevel(VOID);
    };

    std::shared_ptr<tbd::Actor> CreateSphere(CONST util::Vec3& pos, BaseLevel* level);
    VOID CreateStaticPlane(BaseLevel* level);
    std::shared_ptr<tbd::Actor> CreateCube(CONST util::Vec3& pos, BaseLevel* level);
}
