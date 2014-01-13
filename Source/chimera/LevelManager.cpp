#include "Level.h"
#include "Components.h"
#include "tinyxml2.h"
#include "Event.h"

namespace chimera
{
    LevelManager::LevelManager(VOID)
    {
        m_formatSaver[0] = new SaveXMLLevel();
    }

    LevelManager::~LevelManager(VOID)
    {
        TBD_FOR_INT(eCNT)
        {
            SAFE_DELETE(m_formatSaver[i]);
        }
    }

    BOOL SaveXMLLevel::VSaveLevel(ILevel* level, LPCSTR file)
    {
        tinyxml2::XMLDocument doc;

        tinyxml2::XMLDeclaration* version = doc.NewDeclaration("1.0.0");

        doc.LinkEndChild(version);

        tinyxml2::XMLElement* root = doc.NewElement("Level");
        
        CONST std::map<ActorId, std::unique_ptr<IActor>>& actors = level->VGetActors();

        std::vector<ActorId> wasChild;

        for(auto it = actors.begin(); it != actors.end(); ++it)
        {
            IActor* actor = it->second.get();          
            if(!actor->VHasComponent(CM_CMP_PARENT_ACTOR))
            {
                tinyxml2::XMLElement* xmlactor = doc.NewElement("Actor");

                for(auto it2 = actor->VGetComponents().begin(); it2 != actor->VGetComponents().end(); ++it2)
                {
                    //it2->second->VSave(xmlactor);
                }

                ISceneNode* node = CmGetApp()->VGetHumanView()->VGetSceneGraph()->VFindActorNode(actor->GetId());
                if(node)
                {
                    for(auto it3 = node->VGetChilds().begin(); it3 != node->VGetChilds().end(); ++it3)
                    {                    
                        tinyxml2::XMLDocument* doc = xmlactor->GetDocument();
                        tinyxml2::XMLElement* cmp = doc->NewElement("Actor");
                        IActor* child = CmGetApp()->VGetLogic()->VFindActor(it3->get()->VGetActorId());
                        for(auto it4 = child->VGetComponents().begin(); it4!= child->VGetComponents().end(); ++it4)
                        {
                           // it4->second->VSave(cmp);
                        }

                        xmlactor->LinkEndChild(cmp);
                    }
                }

                root->LinkEndChild(xmlactor);
            }
        } 

        doc.LinkEndChild(root);

        std::string s;
        s += CmGetApp()->VGetCache()->VGetFile().VGetName();
        s += "/";
        s += CmGetApp()->VGetConfig()->VGetString("sLevelPath");
        s += file + std::string(".xml");

        return doc.SaveFile(s.c_str());
    }

    BaseLevel::BaseLevel(CONST std::string& file, IActorFactory* factory) : m_file(file), m_name("unnamed"), m_pActorFactory(factory)
    {
        ADD_EVENT_LISTENER(this, &XMLLevel::ActorCreatedDelegate, CM_EVENT_ACTOR_CREATED);
    }

    IActor* BaseLevel::VFindActor(ActorId id)
    {
        auto it = m_actors.find(id);
        if(it == m_actors.end()) 
        {
            return NULL;
        }
        return it->second.get();
    }

    UINT BaseLevel::VGetActorsCount(VOID)
    {
        return (UINT)m_actors.size();
    }

    VOID BaseLevel::VRemoveActor(ActorId id)
    {
        m_actors.erase(id);
    }

    VOID BaseLevel::ActorCreatedDelegate(IEventPtr eventData)
    {
        std::shared_ptr<ActorCreatedEvent> data = std::static_pointer_cast<ActorCreatedEvent>(eventData);
        auto it = std::find(m_idsToLoad.begin(), m_idsToLoad.end(), data->m_id);
        if(it != m_idsToLoad.end())
        {
            m_idsToLoad.erase(it);
            if(m_idsToLoad.empty())
            {
                IEventPtr levelLoadedEvent(new LevelLoadedEvent(std::string(VGetName())));
                QUEUE_EVENT(levelLoadedEvent);
            }
        }
    }

    IActor* BaseLevel::VAddActor(std::unique_ptr<ActorDescription> desc)
    {
        std::unique_ptr<IActor> actor = m_pActorFactory->VCreateActor(std::move(desc));
        IActor* raw = actor.get();
        m_idsToLoad.push_back(actor->GetId());
        m_actors[actor->GetId()] = std::move(actor);
        return raw;
    }

    FLOAT BaseLevel::VGetLoadingProgress(VOID)
    {
        return 1.0f - m_idsToLoad.size() / (FLOAT)VGetActorsCount();
    }

    VOID BaseLevel::VUnload(VOID)
    {
        for(auto it = m_actors.begin(); it != m_actors.end(); ++it)
        {
            IEventPtr deletActorEvent(new DeleteActorEvent(it->first));
            QUEUE_EVENT(deletActorEvent);
        }
    }

    BaseLevel::~BaseLevel(VOID)
    {
        VUnload();
        REMOVE_EVENT_LISTENER(this, &XMLLevel::ActorCreatedDelegate, CM_EVENT_ACTOR_CREATED);
    }

    XMLLevel::XMLLevel(CONST std::string& file, IActorFactory* factory) : BaseLevel(file, factory)
    {

    }

    IActor* XMLLevel::VAddActor(tinyxml2::XMLNode* pNode)
    {
        if(!pNode) 
        {
            LOG_CRITICAL_ERROR("TiXmlElement cant be NULL");
        }
        
        IActor* parent = NULL;
        std::vector<std::unique_ptr<IActor>> actors;
        parent = m_pActorFactory->VCreateActor((ICMStream*)pNode, actors);
        TBD_FOR(actors)
        {
            IActor* actor = it->get();
            m_idsToLoad.push_back(actor->GetId());
            m_actors[actor->GetId()] = std::move(*it);
        }
        return parent;
    }

    BOOL XMLLevel::VSave(LPCSTR file /* = NULL */)
    {

        return TRUE;
    }

    BOOL XMLLevel::VLoad(BOOL block)
    {
        tinyxml2::XMLDocument doc;

        CMResource r(CmGetApp()->VGetConfig()->VGetString("sLevelPath") + VGetFile().c_str());

        std::shared_ptr<IResHandle> handle = CmGetApp()->VGetCache()->VGetHandle(r);

        if(!handle)
        {
            return FALSE;
        }

        doc.Parse((CHAR*)handle->VBuffer());

        tinyxml2::XMLElement* root = doc.RootElement();
        RETURN_IF_FAILED(root);

        VAddActor(root);

        return TRUE;
    }
}