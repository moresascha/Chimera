#include "Process.h"
#include "GameApp.h"
#include "Actor.h"
#include "EventManager.h"
#include "ShaderProgram.h"
#include "d3d.h"
#include "Components.h"
#include "Camera.h"
#include "Cudah.h"
#include "Commands.h"
#include "util.h"
#include <strsafe.h>

namespace proc 
{
    std::shared_ptr<Process> Process::RemoveChild(VOID) {
        if(m_child)
        {
            std::shared_ptr<Process> ptr = m_child;
            m_child.reset();
            return ptr;
        }

        return std::shared_ptr<Process>();
    }

    ActorProcess::ActorProcess(std::shared_ptr<tbd::Actor> actor) : m_actor(actor), m_isCreated(FALSE)
    {

    }

    VOID ActorProcess::ActorCreatedDelegate(event::IEventPtr data)
    {
        std::shared_ptr<event::ActorCreatedEvent> event = std::static_pointer_cast<event::ActorCreatedEvent>(data);
        if(event->m_id == m_actor->GetId())
        {
            m_isCreated = TRUE;
            VOnActorCreate();
        }
    }

    VOID ActorProcess::DeleteActorDelegate(event::IEventPtr data)
    {
        std::shared_ptr<event::DeleteActorEvent> event = std::static_pointer_cast<event::DeleteActorEvent>(data);
        if(event->m_id == m_actor->GetId())
        {
            VOnActorDelete();
            Succeed();
        }
    }

    VOID ActorProcess::VOnInit(VOID)
    {
        if(m_actor)
        {
            event::EventListener listener = fastdelegate::MakeDelegate(this, &ActorProcess::ActorCreatedDelegate);
            event::IEventManager::Get()->VAddEventListener(listener, event::ActorCreatedEvent::TYPE);

            listener = fastdelegate::MakeDelegate(this, &ActorProcess::DeleteActorDelegate);
            event::IEventManager::Get()->VAddEventListener(listener, event::DeleteActorEvent::TYPE);
        }
    }

    ActorProcess::~ActorProcess(VOID)
    {
        if(m_actor)
        {
            event::EventListener listener = fastdelegate::MakeDelegate(this, &ActorProcess::ActorCreatedDelegate);
            event::IEventManager::Get()->VRemoveEventListener(listener, event::ActorCreatedEvent::TYPE);

            listener = fastdelegate::MakeDelegate(this, &ActorProcess::DeleteActorDelegate);
            event::IEventManager::Get()->VRemoveEventListener(listener, event::DeleteActorEvent::TYPE);
        }
    }

    RealtimeProcess::RealtimeProcess(int priority) : m_threadId(0), m_priority(priority), m_pThreadHandle(0) { }

    VOID RealtimeProcess::VOnInit(VOID) 
    {
    
        //Process::VOnInit();
        m_pWaitHandle = CreateEvent(NULL, FALSE, FALSE, NULL);
        this->m_pThreadHandle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(ThreadProc), this, 0, &this->m_threadId);
    
        if(!this->m_pThreadHandle)
        {
            this->Fail();
            LOG_CRITICAL_ERROR("Thread creation failed!");
            return;
        }

        SetThreadPriority(this->m_pThreadHandle, this->m_priority);
    }

    DWORD WINAPI RealtimeProcess::ThreadProc(LPVOID lpParam) 
    {
        RealtimeProcess* pP = static_cast<RealtimeProcess*>(lpParam);
        pP->VThreadProc();
        SetEvent(pP->m_pWaitHandle);
        return S_OK;
    }

    VOID RealtimeProcess::WaitComplete(VOID)
    {
        if(m_pThreadHandle)
        {
            WaitForSingleObject(m_pWaitHandle, INFINITE);
        }
    }

    VOID RealtimeProcess::VOnAbort(VOID)
    { 
        WaitComplete();
    }

    VOID RealtimeProcess::VOnSuccess(VOID)
    { 
        WaitComplete();
    }

    VOID RealtimeProcess::VOnFail(VOID)
    { 
        WaitComplete();
    }

    RealtimeProcess::~RealtimeProcess(VOID)
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

    ActorRealtimeProcess::ActorRealtimeProcess(std::shared_ptr<tbd::Actor> actor) : m_actor(actor), m_event(NULL)
    {
        event::EventListener listener = fastdelegate::MakeDelegate(this, &ActorRealtimeProcess::ActorCreatedDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::ActorCreatedEvent::TYPE);
    
        listener = fastdelegate::MakeDelegate(this, &ActorRealtimeProcess::DeleteActorDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::DeleteActorEvent::TYPE);

        m_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    }

    VOID ActorRealtimeProcess::VThreadProc(VOID)
    {
        WaitForSingleObject(m_event, INFINITE);
        _VThreadProc();
    }

    VOID ActorRealtimeProcess::ActorCreatedDelegate(event::IEventPtr data)
    {
        std::shared_ptr<event::ActorCreatedEvent> event = std::static_pointer_cast<event::ActorCreatedEvent>(data);
        if(event->m_id == m_actor->GetId())
        {
            SetEvent(m_event);
        }
    }

    VOID ActorRealtimeProcess::DeleteActorDelegate(event::IEventPtr data)
    {
        std::shared_ptr<event::DeleteActorEvent> event = std::static_pointer_cast<event::DeleteActorEvent>(data);
        if(event->m_id == m_actor->GetId())
        {
            Succeed();
            SetEvent(m_event);
        }
    }

    ActorRealtimeProcess::~ActorRealtimeProcess(VOID)
    {
        event::EventListener listener = fastdelegate::MakeDelegate(this, &ActorRealtimeProcess::ActorCreatedDelegate);
        event::IEventManager::Get()->VRemoveEventListener(listener, event::ActorCreatedEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &ActorRealtimeProcess::DeleteActorDelegate);
        event::IEventManager::Get()->VRemoveEventListener(listener, event::DeleteActorEvent::TYPE);
        
        if(m_event)
        {
            CloseHandle(m_event);
        }
    }

    VOID RotationProcess::_VThreadProc(VOID)
    {
        ActorId id = m_actor->GetId();
        while(!IsDead())
        {
            event::IEventPtr moveActor(new event::MoveActorEvent(id, util::Vec3(), m_rotations, TRUE));
            event::IEventManager::Get()->VQueueEventThreadSave(moveActor);
            Sleep(32);
        }
    }

    StrobeLightProcess::StrobeLightProcess(std::shared_ptr<tbd::LightComponent> lightComponent, FLOAT prob, UINT freq) : m_lightComponent(lightComponent), m_prob(prob), m_freq(freq)
    {
    }

    VOID StrobeLightProcess::VThreadProc()
    {
        while(IsAlive())
        {
            Sleep(m_freq);
            m_lightComponent->m_activated = rand() / (FLOAT)RAND_MAX < m_prob;
        }
    }

    //file modification
    WatchDirModifacationProcess::WatchDirModifacationProcess(LPCTSTR dir) : m_dir(dir), m_fileHandle(NULL), m_closeHandle(NULL)
    {

    }

    VOID WatchDirModifacationProcess::VThreadProc(VOID)
    {
        HANDLE handles[2];
        handles[0] = m_fileHandle;
        handles[1] = m_closeHandle;
        while(IsAlive())
        {
            DWORD event = WaitForMultipleObjects(2, handles, FALSE, INFINITE);
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

    VOID WatchDirModifacationProcess::VOnInit(VOID)
    {
        std::wstring s(m_dir.begin(), m_dir.end());
        m_fileHandle = FindFirstChangeNotification(s.c_str(), FALSE, FILE_NOTIFY_CHANGE_LAST_WRITE);
        if(INVALID_HANDLE_VALUE == m_fileHandle)
        {
            LOG_CRITICAL_ERROR("FindFirstChangeNotification failed hard");
        }
        m_closeHandle = CreateEvent(NULL, FALSE, FALSE, NULL);

        RealtimeProcess::VOnInit();
    }

    VOID WatchDirModifacationProcess::Close(VOID)
    {
        if(m_fileHandle)
        {
            SetEvent(m_closeHandle);
        }
    }

    VOID WatchDirModifacationProcess::VOnAbort(VOID)
    {
        Close();
        RealtimeProcess::VOnAbort();
    }

    VOID WatchDirModifacationProcess::VOnFail(VOID)
    {
        Close();
        RealtimeProcess::VOnFail();
    }

    VOID WatchDirModifacationProcess::VOnSuccess(VOID)
    {
        Close();
        RealtimeProcess::VOnSuccess();
    }

    WatchDirModifacationProcess::~WatchDirModifacationProcess(VOID)
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

    WatchFileModificationProcess::WatchFileModificationProcess(LPCTSTR file, LPCTSTR dir) : WatchDirModifacationProcess(dir), m_file(file), m_update(FALSE)
    {

    }

    VOID WatchFileModificationProcess::VOnInit(VOID)
    {
        m_lastModification = GetTime();

        WatchDirModifacationProcess::VOnInit();
    }

    VOID WatchFileModificationProcess::VOnFail(VOID)
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

    time_t WatchFileModificationProcess::GetTime(VOID)
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

    VOID WatchFileModificationProcess::VOnDirModification(VOID)
    {
        time_t time = GetTime();
        if(m_lastModification < time)
        {
            m_update = TRUE;
            m_lastModification = time;
        }
    }

    VOID WatchFileModificationProcess::VOnUpdate(ULONG deltaMillis)
    {
        if(m_update)
        {
            VOnFileModification();
        }
        m_update = FALSE;
    }

    WatchShaderFileModificationProcess::WatchShaderFileModificationProcess(d3d::Shader* shader, LPCTSTR file, LPCTSTR dir) : WatchFileModificationProcess(file, dir), m_shader(shader)
    {

    }

    VOID WatchShaderFileModificationProcess::VOnFileModification(VOID)
    {
        d3d::ErrorLog log;
        if(!m_shader->Compile(&log))
        {
            DEBUG_OUT_A("Failed to compile, no changes were made.\n %s", log.c_str());
        }
    }

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
        std::string path = app::g_pApp->GetConfig()->GetString("sPTXPath");
        sprintf_s(buffer, "runproc nvcc -arch=sm_10 --use-local-env --cl-version 2010 -I\"C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\include\" -I\"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v4.2\\include\" -I\"../include\" -G --keep --keep-dir \"E:\\Dropbox\\Visual Studio\\Chimera\\Source\\..\\Tmp\\Chimerax64Debug\" -maxrregcount=0  --machine 64 -ptx  -o \"%s%s\" \"E:\\Dropbox\\Visual Studio\\Chimera\\Source\\chimera\\%s\"", path.c_str(), fn.c_str(), fff.c_str());
        app::g_pApp->GetLogic()->GetCommandInterpreter()->CallCommand(buffer);
        m_pCuda->OnRestore();
    }
};
