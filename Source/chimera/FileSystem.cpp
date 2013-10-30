#include "FileSystem.h"
#include "Process.h"
#include "EventManager.h"
namespace chimera
{
    bool operator<(CONST _File o, CONST _File o1)
    {
        return o.m_name < o.m_name;
    }

    DLL::DLL(LPCSTR path)
    {
        m_instance = NULL;
        m_path = path;

        std::string pf = m_path;
        m_instance = LoadLibraryA(pf.c_str());
        if(m_instance == NULL)
        {
            LOG_CRITICAL_ERROR("Failed to load library");
        }
    }

    DLL::~DLL(VOID)
    {
        if(m_instance)
        {
            FreeLibrary(m_instance); 
            m_instance = NULL;
        }
    }

    FileSystem::FileSystem(void)
    {
    }

    VOID FileSystem::RegisterCallback(LPCSTR fileName, LPCSTR path, OnFileChangedCallback cb)
    {
        _File f(fileName);
        auto it = m_callBackMap.find(f);
        if(it == m_callBackMap.end())
        {
            f.m_path = path;
            f.m_fullPath = path;
            f.m_fullPath += fileName;
            std::list<OnFileChangedCallback>* list = new std::list<OnFileChangedCallback>();
            m_callBackMap.insert(std::pair<_File, std::list<OnFileChangedCallback>*>(f, list));
            chimera::CreateProcessEvent* cpe = new chimera::CreateProcessEvent();
            cpe->m_pProcess = new chimera::FileWatcherProcess(fileName, path);
            QUEUE_EVENT(cpe);
            ADD_EVENT_LISTENER(this, &chimera::FileSystem::OnFileChanged, chimera::FileChangedEvent::TYPE);
        }
        std::list<OnFileChangedCallback>* list = m_callBackMap.find(f)->second;
        auto itt = std::find(list->begin(), list->end(), cb);

        if(itt == list->end())
        {
            m_callBackMap.find(f)->second->push_back(cb);
        }
    }

    VOID FileSystem::OnFileChanged(std::shared_ptr<chimera::IEvent> data)
    {
        std::shared_ptr<chimera::FileChangedEvent> fce = std::static_pointer_cast<chimera::FileChangedEvent>(data);
        auto list = m_callBackMap.find(fce->m_file);
        if(list != m_callBackMap.end())
        {
            for(auto it = list->second->begin(); it != list->second->end(); ++it)
            {
                (*it)();
            }
        }
    }

    FileSystem::~FileSystem(void)
    {
        REMOVE_EVENT_LISTENER(this, &chimera::FileSystem::OnFileChanged, chimera::FileChangedEvent::TYPE);


        TBD_FOR(m_callBackMap)
        {
            SAFE_DELETE(it->second);
        }
    }
};
