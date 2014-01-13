#pragma once
#include "stdafx.h"

namespace tinyxml2
{
    class XMLNode;
}

namespace chimera
{
    class ISaveXLevel
    {
    public:
        virtual BOOL VSaveLevel(ILevel* level, LPCSTR file) = 0;
    };

    class SaveXMLLevel : public ISaveXLevel
    {
    public:
        BOOL VSaveLevel(ILevel* level, LPCSTR file);
    };

    class LevelManager
    {
    private:
        ISaveXLevel* m_formatSaver[eCNT];
    public:
        LevelManager(VOID);
        ~LevelManager(VOID);
    };

    class BaseLevel : public ILevel
    {
    private:
        std::string m_name;
        std::string m_file;
        
    protected:
        std::map<ActorId, std::unique_ptr<IActor>> m_actors;
        std::vector<ActorId> m_idsToLoad;
        IActorFactory* m_pActorFactory;

    public:

        BaseLevel::BaseLevel(CONST std::string& file, IActorFactory* factory);

        CONST std::string& VGetName(VOID)
        {
            return m_name;
        }

        CONST std::string& VGetFile(VOID)
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

        CONST std::map<ActorId, std::unique_ptr<IActor>>& VGetActors(VOID)
        {
            return m_actors;
        }

        UINT VGetActorsCount(VOID);

        IActor* VFindActor(ActorId id);

        VOID VRemoveActor(ActorId id);

        VOID ActorCreatedDelegate(IEventPtr eventData);

        FLOAT VGetLoadingProgress(VOID);

        VOID VUnload(VOID);

        IActor* VAddActor(std::unique_ptr<ActorDescription> desc);

        virtual ~BaseLevel(VOID);
    };

    class XMLLevel : public BaseLevel
    {
    private:

    public:
        XMLLevel(CONST std::string& file, IActorFactory* factory);
        
        BOOL VLoad(BOOL block);
        
        BOOL VSave(LPCSTR file = NULL);
                
        IActor* VAddActor(tinyxml2::XMLNode* pNode);
    };

    class RandomLevel : public BaseLevel
    {
    public:
        RandomLevel(CONST std::string& file, IActorFactory* factory);

        BOOL VLoad(BOOL block);

        BOOL VSave(LPCSTR file = NULL);
    };

    class GroupedObjLevel : public BaseLevel
    {
    public:
        GroupedObjLevel(LPCSTR file, IActorFactory* factory);
        BOOL VLoad(BOOL block);
        BOOL VSave(LPCSTR file = NULL);
    };

    class TransformShowRoom : public BaseLevel
    {
    public:
        TransformShowRoom(CONST std::string& file, IActorFactory* factory);

        BOOL VLoad(BOOL block);

        BOOL VSave(LPCSTR file = NULL);
    };

    class CudaTransformationNode;

    class BSplinePatchLevel : public BaseLevel
    {
    private:
        IActor* m_controlPoints;
        chimera::CudaTransformationNode* m_node;

    public:
        BSplinePatchLevel(CONST std::string& file, IActorFactory* factory);

        BOOL VLoad(BOOL block);

        BOOL VSave(LPCSTR file = NULL);

        VOID OnControlPointMove(chimera::IEventPtr data);

        ~BSplinePatchLevel(VOID);
    };

    //std::shared_ptr<IActor> CreateSphere(CONST util::Vec3& pos, BaseLevel* level);
    //VOID CreateStaticPlane(BaseLevel* level);
    //std::shared_ptr<chimera::Actor> CreateCube(CONST util::Vec3& pos, BaseLevel* level);
}
