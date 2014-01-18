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
        virtual bool VSaveLevel(ILevel* level, LPCSTR file) = 0;
    };

    class SaveXMLLevel : public ISaveXLevel
    {
    public:
        bool VSaveLevel(ILevel* level, LPCSTR file);
    };

    class LevelManager
    {
    private:
        ISaveXLevel* m_formatSaver[eCNT];
    public:
        LevelManager(void);
        ~LevelManager(void);
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

        BaseLevel::BaseLevel(const std::string& file, IActorFactory* factory);

        const std::string& VGetName(void)
        {
            return m_name;
        }

        const std::string& VGetFile(void)
        {
            return m_file;
        }

        void SetFile(const std::string& file)
        {
            m_file = file;
        }

        void SetName(const std::string& name)
        {
            m_name = name;
        }

        const std::map<ActorId, std::unique_ptr<IActor>>& VGetActors(void)
        {
            return m_actors;
        }

        uint VGetActorsCount(void);

        IActor* VFindActor(ActorId id);

        void VRemoveActor(ActorId id);

        void ActorCreatedDelegate(IEventPtr eventData);

        float VGetLoadingProgress(void);

        void VUnload(void);

        IActor* VAddActor(std::unique_ptr<ActorDescription> desc);

        virtual ~BaseLevel(void);
    };

    class XMLLevel : public BaseLevel
    {
    private:

    public:
        XMLLevel(const std::string& file, IActorFactory* factory);
        
        bool VLoad(bool block);
        
        bool VSave(LPCSTR file = NULL);
                
        IActor* VAddActor(tinyxml2::XMLNode* pNode);
    };

    class RandomLevel : public BaseLevel
    {
    public:
        RandomLevel(const std::string& file, IActorFactory* factory);

        bool VLoad(bool block);

        bool VSave(LPCSTR file = NULL);
    };

    class GroupedObjLevel : public BaseLevel
    {
    public:
        GroupedObjLevel(LPCSTR file, IActorFactory* factory);
        bool VLoad(bool block);
        bool VSave(LPCSTR file = NULL);
    };

    class TransformShowRoom : public BaseLevel
    {
    public:
        TransformShowRoom(const std::string& file, IActorFactory* factory);

        bool VLoad(bool block);

        bool VSave(LPCSTR file = NULL);
    };

    class CudaTransformationNode;

    class BSplinePatchLevel : public BaseLevel
    {
    private:
        IActor* m_controlPoints;
        chimera::CudaTransformationNode* m_node;

    public:
        BSplinePatchLevel(const std::string& file, IActorFactory* factory);

        bool VLoad(bool block);

        bool VSave(LPCSTR file = NULL);

        void OnControlPointMove(chimera::IEventPtr data);

        ~BSplinePatchLevel(void);
    };

    //std::shared_ptr<IActor> CreateSphere(CONST util::Vec3& pos, BaseLevel* level);
    //VOID CreateStaticPlane(BaseLevel* level);
    //std::shared_ptr<chimera::Actor> CreateCube(CONST util::Vec3& pos, BaseLevel* level);
}
