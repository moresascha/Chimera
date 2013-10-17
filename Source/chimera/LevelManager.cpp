#include "Level.h"
#include "tinyxml2.h"
#include "GameApp.h"
#include "Resources.h"
#include "Components.h"
#include "GameView.h"
#include "SceneGraph.h"
#include "GameLogic.h"

namespace chimera
{
    LevelManager::LevelManager(VOID)
    {
        m_formatSaver[0] = new chimera::SaveXMLLevel();
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
        
        std::map<ActorId, std::shared_ptr<chimera::Actor>>& actors = level->VGetActors();

        std::vector<ActorId> wasChild;

        for(auto it = actors.begin(); it != actors.end(); ++it)
        {
            chimera::Actor* actor = it->second.get();          
            if(!actor->HasComponent<chimera::ParentComponent>(chimera::ParentComponent::COMPONENT_ID))
            {
                tinyxml2::XMLElement* xmlactor = doc.NewElement("Actor");

                for(auto it2 = actor->GetComponents().begin(); it2 != actor->GetComponents().end(); ++it2)
                {
                    it2->second->VSave(xmlactor);
                }
                std::shared_ptr<ISceneNode> node = chimera::g_pApp->GetHumanView()->GetSceneGraph()->FindActorNode(actor->GetId());
				if(node)
				{
					for(auto it3 = node->GetChilds().begin(); it3 != node->GetChilds().end(); ++it3)
					{                    
						tinyxml2::XMLDocument* doc = xmlactor->GetDocument();
						tinyxml2::XMLElement* cmp = doc->NewElement("Actor");
						std::shared_ptr<chimera::Actor> child = chimera::g_pApp->GetLogic()->VFindActor(it3->get()->VGetActorId());
						for(auto it4 = child->GetComponents().begin(); it4!= child->GetComponents().end(); ++it4)
						{
							it4->second->VSave(cmp);
						}

						xmlactor->LinkEndChild(cmp);
                    }
                }

                root->LinkEndChild(xmlactor);
            }
        } 

        doc.LinkEndChild(root);

        std::string s;
        s += chimera::g_pApp->GetCache()->GetFile().VGetName();
        s += "/";
        s += chimera::g_pApp->GetConfig()->GetString("sLevelPath");
        s += file + std::string(".xml");

        return doc.SaveFile(s.c_str());
    }
}