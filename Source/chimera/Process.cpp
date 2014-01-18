#include "Process.h"
#include "Event.h"
#include <time.h>

namespace chimera 
{
    ActorProcess::ActorProcess(IActor* actor) : m_actor(actor), m_isCreated(false)
    {

    }

    void ActorProcess::ActorCreatedDelegate(IEventPtr data)
    {
        std::shared_ptr<ActorCreatedEvent> event = std::static_pointer_cast<ActorCreatedEvent>(data);
        if(event->m_id == m_actor->GetId())
        {
            m_isCreated = true;
            VOnActorCreate();
        }
    }

    void ActorProcess::DeleteActorDelegate(IEventPtr data)
    {
        std::shared_ptr<DeleteActorEvent> event = std::static_pointer_cast<DeleteActorEvent>(data);
        if(event->m_id == m_actor->GetId())
        {
            VOnActorDelete();
            Succeed();
        }
    }

    void ActorProcess::VOnInit(void)
    {
        if(m_actor)
        {
            ADD_EVENT_LISTENER(this, &ActorProcess::ActorCreatedDelegate, CM_EVENT_ACTOR_CREATED);
            ADD_EVENT_LISTENER(this, &ActorProcess::DeleteActorDelegate, CM_EVENT_DELETE_ACTOR);
        }
    }

    ActorProcess::~ActorProcess(void)
    {
        if(m_actor)
        {
            REMOVE_EVENT_LISTENER(this, &ActorProcess::ActorCreatedDelegate, CM_EVENT_ACTOR_CREATED);
            REMOVE_EVENT_LISTENER(this, &ActorProcess::DeleteActorDelegate, CM_EVENT_DELETE_ACTOR);
        }
    }

    RealtimeProcess::RealtimeProcess(int priority) : m_threadId(0), m_priority(priority), m_pThreadHandle(NULL), m_pWaitHandle(NULL) { }

    void RealtimeProcess::VOnInit(void) 
    {
        //Process::VOnInit();
        m_pWaitHandle = CreateEvent(NULL, false, false, NULL);
        m_pThreadHandle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(ThreadProc), this, 0, &this->m_threadId);
    
        if(!this->m_pThreadHandle)
        {
            Fail();
            LOG_CRITICAL_ERROR("Thread creation failed!");
            return;
        }

        SetThreadPriority(m_pThreadHandle, this->m_priority);
    }

    DWORD WINAPI RealtimeProcess::ThreadProc(LPVOID lpParam) 
    {
        RealtimeProcess* pP = static_cast<RealtimeProcess*>(lpParam);
        pP->VThreadProc();
        SetEvent(pP->m_pWaitHandle);
        return S_OK;
    }

    void RealtimeProcess::WaitComplete(void)
    {
        if(m_pThreadHandle)
        {
            WaitForSingleObject(m_pWaitHandle, INFINITE);
        }
    }

    void RealtimeProcess::VOnAbort(void)
    { 
        WaitComplete();
    }

    void RealtimeProcess::VOnSuccess(void)
    { 
        WaitComplete();
    }

    void RealtimeProcess::VOnFail(void)
    { 
        WaitComplete();
    }

    RealtimeProcess::~RealtimeProcess(void)
    { 
        if(m_pThreadHandle) 
        {
            CloseHandle(m_pThreadHandle);
            //DEBUG_OUT_A("closing thread...%d", m_hThread);
        }
        if(m_pWaitHandle)
        {
            CloseHandle(m_pWaitHandle);
        }
    }

    ActorRealtimeProcess::ActorRealtimeProcess(IActor* actor) : m_actor(actor), m_event(NULL)
    {

        ADD_EVENT_LISTENER(this, &ActorRealtimeProcess::ActorCreatedDelegate, CM_EVENT_ACTOR_CREATED);
        ADD_EVENT_LISTENER(this, &ActorRealtimeProcess::DeleteActorDelegate, CM_EVENT_DELETE_ACTOR);
    
        m_event = CreateEvent(NULL, false, false, NULL);
    }

    void ActorRealtimeProcess::VThreadProc(void)
    {
        WaitForSingleObject(m_event, INFINITE);
        _VThreadProc();
    }

    void ActorRealtimeProcess::ActorCreatedDelegate(IEventPtr data)
    {
        std::shared_ptr<ActorCreatedEvent> event = std::static_pointer_cast<ActorCreatedEvent>(data);
        if(event->m_id == m_actor->GetId())
        {
            SetEvent(m_event);
        }
    }

    void ActorRealtimeProcess::DeleteActorDelegate(IEventPtr data)
    {
        std::shared_ptr<DeleteActorEvent> event = std::static_pointer_cast<DeleteActorEvent>(data);
        if(event->m_id == m_actor->GetId())
        {
            Succeed();
            SetEvent(m_event);
        }
    }

    ActorRealtimeProcess::~ActorRealtimeProcess(void)
    {
        REMOVE_EVENT_LISTENER(this, &ActorRealtimeProcess::ActorCreatedDelegate, CM_EVENT_ACTOR_CREATED);
        REMOVE_EVENT_LISTENER(this, &ActorRealtimeProcess::DeleteActorDelegate, CM_EVENT_DELETE_ACTOR);
        
        if(m_event)
        {
            CloseHandle(m_event);
        }
    }

    //file modification
    WatchDirModifacationProcess::WatchDirModifacationProcess(LPCTSTR dir) : m_dir(dir), m_fileHandle(NULL), m_closeHandle(NULL)
    {

    }

    void WatchDirModifacationProcess::VThreadProc(void)
    {
        HANDLE handles[2];
        handles[0] = m_fileHandle;
        handles[1] = m_closeHandle;
        while(IsAlive())
        {
            DWORD event = WaitForMultipleObjects(2, handles, false, INFINITE);
            if(event == WAIT_FAILED)
            {
                Fail();
                return;
            }
            if(IsAlive() && (event == WAIT_OBJECT_0 || (event == (WAIT_OBJECT_0 + 1))))
            {
                VOnDirModification();
                FindNextChangeNotification(m_fileHandle);
            }
        }
    }

    void WatchDirModifacationProcess::VOnInit(void)
    {
        std::wstring s(m_dir.begin(), m_dir.end());
        m_fileHandle = FindFirstChangeNotification(s.c_str(), false, FILE_NOTIFY_CHANGE_LAST_WRITE);
        if(INVALID_HANDLE_VALUE == m_fileHandle)
        {
            LOG_CRITICAL_ERROR("FindFirstChangeNotification failed hard");
        }
        m_closeHandle = CreateEvent(NULL, false, false, NULL);

        RealtimeProcess::VOnInit();
    }

    void WatchDirModifacationProcess::Close(void)
    {
        if(m_fileHandle)
        {
            SetEvent(m_closeHandle);
        }
    }

    void WatchDirModifacationProcess::VOnAbort(void)
    {
        Close();
        RealtimeProcess::VOnAbort();
    }

    void WatchDirModifacationProcess::VOnFail(void)
    {
        Close();
        RealtimeProcess::VOnFail();
    }

    void WatchDirModifacationProcess::VOnSuccess(void)
    {
        Close();
        RealtimeProcess::VOnSuccess();
    }

    WatchDirModifacationProcess::~WatchDirModifacationProcess(void)
    {
        if(m_fileHandle)
        {
            FindCloseChangeNotification(m_fileHandle);
        }
        if(m_closeHandle)
        {
            CloseHandle(m_closeHandle);
        }
    }

    WatchFileModificationProcess::WatchFileModificationProcess(LPCTSTR file, LPCTSTR dir) : WatchDirModifacationProcess(dir), m_file(file), m_update(false)
    {

    }

    void WatchFileModificationProcess::VOnInit(void)
    {
        m_lastModification = GetTime();

        WatchDirModifacationProcess::VOnInit();
    }

    void WatchFileModificationProcess::VOnFail(void)
    {
#ifdef _DEBUG
        std::wstring ws;
        ws = m_dir + m_file;
        std::string ns(ws.begin(), ws.end());
        std::string wsd(m_dir.begin(), m_dir.end());
        DEBUG_OUT_A("WatchFileModificationProcess failed, closing WatcherThread for File %s, in Directory %s\n", ns.c_str(), wsd.c_str());
#endif
        WatchDirModifacationProcess::VOnFail();
    }

    time_t WatchFileModificationProcess::GetTime(void)
    {
        std::wstring s = m_dir + m_file;
        HANDLE file = CreateFile(s.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
            OPEN_EXISTING, 0, NULL);

        if(file)
        {
            FILETIME ftCreate, ftAccess, ftWrite;
            SYSTEMTIME writeTime;    

            GetFileTime(file, &ftCreate, &ftAccess, &ftWrite);

            FileTimeToSystemTime(&ftWrite, &writeTime);

            //FileTimeToSystemTime(&ftCreate, &createTime);

            //SystemTimeToTzSpecificLocalTime(NULL, &stUTC, &stLocal);

            struct tm tmTime;
            tmTime.tm_hour = writeTime.wHour;
            tmTime.tm_min = writeTime.wMinute;
            tmTime.tm_mday = writeTime.wDay;
            tmTime.tm_mon = writeTime.wMonth;
            tmTime.tm_sec = writeTime.wSecond;
            tmTime.tm_year = writeTime.wYear - 1900;

            time_t t = mktime(&tmTime);

            CloseHandle(file);

            return t;
        }
        else
        {
            LOG_CRITICAL_ERROR("File not found");
            return -1;
        }
    }

    void WatchFileModificationProcess::VOnDirModification(void)
    {
        time_t time = GetTime();
        if(m_lastModification < time)
        {
            m_update = true;
            m_lastModification = time;
        }
    }

    void WatchFileModificationProcess::VOnUpdate(ulong deltaMillis)
    {
        if(m_update)
        {
            VOnFileModification();
        }
        m_update = false;
    }

    FileWatcherProcess::FileWatcherProcess(LPCTSTR file, LPCTSTR dir) : chimera::WatchFileModificationProcess(file, dir)
    {

    }

    FileWatcherProcess::FileWatcherProcess(LPCSTR file, LPCSTR dir) : chimera::WatchFileModificationProcess(L"", L"")
    {
        std::string f(file);
        std::string d(dir);
        m_file = std::wstring(f.begin(), f.end());
        m_dir = std::wstring(d.begin(), d.end());
    }

    void FileWatcherProcess::VOnFileModification(void)
    {
        QUEUE_EVENT_TSAVE(new chimera::FileChangedEvent(m_file.c_str()));
    }

    WatchShaderFileModificationProcess::WatchShaderFileModificationProcess(IShader* shader, LPCTSTR file, LPCTSTR dir) : WatchFileModificationProcess(file, dir), m_shader(shader)
    {

    }

    void WatchShaderFileModificationProcess::VOnFileModification(void)
    {
        chimera::ErrorLog log;
        if(!m_shader->VCompile(&log))
        {
            DEBUG_OUT_A("Failed to compile, no changes were made.\n %s", log.c_str());
        }
    }

    /*
    WatchCudaFileModificationProcess::WatchCudaFileModificationProcess(cudah::cudah* cuda, LPCTSTR file, LPCTSTR dir) : WatchFileModificationProcess(file, dir), m_pCuda(cuda)
    {

    }

    VOID WatchCudaFileModificationProcess::VOnFileModification(VOID)
    {
        char buffer[8000];
        std::string ff(m_file.begin(), m_file.end());
        std::string fff(m_file.begin(), m_file.end());
        std::string fn = util::split(ff, '.')[0];
        fn += ".ptx";
        std::string path = chimera::g_pApp->GetConfig()->GetString("sPTXPath");
        sprintf_s(buffer, "runproc nvcc -arch=sm_20 --use-local-env --cl-version 2012 -I\"C:\\Program Files (x86)\\Microsoft Visual Studio 11.0\\VC\\include\" -I\"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v5.5\\include\" -I\"../include\" -G --keep -maxrregcount=0  --machine 64 -ptx -cudart static -o \"%s%s\" \"E:\\Dropbox\\VisualStudio\\Chimera\\Source\\chimera\\%s\"", path.c_str(), fn.c_str(), fff.c_str());
        chimera::g_pApp->GetLogic()->GetCommandInterpreter()->CallCommand(buffer);
        m_pCuda->OnRestore();
    } */
};
