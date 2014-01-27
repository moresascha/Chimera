#include "Commands.h"
#include "util.h"
#include <fstream>
#include "Event.h"
#include "Components.h"
#include "GuiComponent.h"

namespace chimera
{

    bool CheckForError(std::list<std::string>& toCheck)
    {
        return toCheck.size() == 0;
    }

#define CHECK_FOR_ERROR(_RET) \
    if(CheckForError(m_values)) \
    { \
    m_error = TRUE; \
    return _RET; \
    }

    Command::Command(std::list<std::string>& elems) : m_values(elems), m_error(false)
    {
        
    }

    bool Command::VInitArgumentTypes(int args, ...)
    {
        va_list pointer;

        va_start(pointer, args);

        for(int i = 0; i < args; ++i)
        {
            int format = va_arg(pointer, int);
            m_argList.push_back(format);
        }

        va_end(pointer);

        return true;
    }

    std::string Command::VGetRemainingString(void)
    {
        std::string s;
        while(!m_values.empty())
        {
            s += m_values.front();
            if(m_values.size() > 1)
            {
                s += " ";
            }
            m_values.pop_front();
        }
        return s;
    }

    float Command::VGetNextFloat(void)
    {
        CHECK_FOR_ERROR((float)0.0f);

        char* ptr = NULL;
        const char* toConvert = m_values.begin()->c_str();

        float v = (float)strtod(toConvert, &ptr);

        m_error = ptr && ptr == toConvert;

        m_values.pop_front();
        return v;
    }

    bool Command::VGetNextBool(void)
    {
        m_error = CheckForError(m_values);
        return VGetNextInt() != 0;
    }

    int Command::VGetNextInt(void)
    {
        CHECK_FOR_ERROR((int)0);

        char* ptr = NULL;
        const char* toConvert = m_values.begin()->c_str();

        int v = (int)strtol(toConvert, &ptr, 10);

        m_error = ptr && ptr == toConvert;

        m_values.pop_front();
        return v;
    }

    char Command::VGetNextChar(void)
    {
        CHECK_FOR_ERROR((char)0);

        std::string s = *m_values.begin();
        m_values.pop_front();
        return (char)s.c_str()[0];
    }

    std::string Command::VGetNextCharStr(void)
    {
        CHECK_FOR_ERROR(std::string());

        std::string s = *m_values.begin();
        m_values.pop_front();
        return s;
    }

    bool Command::VIsError(void)
    {
        return m_error;
    }

    bool Command::VIsValid(void)
    {
        return !VIsError();
    }

    Command::~Command(void)
    {

    }

    CommandInterpreter::CommandInterpreter(void)
    {
        commands::AddCommandsToInterpreter(*this);
    }

    void CommandInterpreter::VRegisterCommand(LPCSTR name, CommandHandler command, LPCSTR usage)
    {
        m_nameToCommandHandler[name] = command;
        if(usage)
        {
            m_nameToUsage[name] = std::string("Invalid command usage: ") + std::string(usage);
        }
        else
        {
            m_nameToUsage[name] = std::string("Invalid command usage: No Usage available.");
        }
    }

    std::vector<std::string> CommandInterpreter::VGetCommands(void)
    {
        std::vector<std::string> list;
        TBD_FOR(m_nameToCommandHandler)
        {
            list.push_back(it->first);
        }
        return list;
    }

    bool CommandInterpreter::VCallCommand(LPCSTR command)
    {
        std::vector<std::string> elements = util::split(std::string(command), ' ');
        if(elements.size() == 0)
        {
            return false;
        }
        //CHECK_VETOR_SIZE(elements);

        std::string& cmd = elements.front();

        auto it = m_nameToCommandHandler.find(std::string(cmd));

        if(it == m_nameToCommandHandler.end())
        {
            //DEBUG_OUT("no commandhandler for command: " + cmd);
            return false;
        }

        std::list<std::string> vals;

        for(int i = 1; i < elements.size(); ++i)
        {
            vals.push_back(elements[i]);
        }

        chimera::Command c(vals);
        try 
        {
            if(!it->second(c))
            {
                std::string printStr("print " + m_nameToUsage[cmd]);
                VCallCommand(printStr.c_str());
            }
        } catch(LPCSTR error)
        {
            std::string printStr("print ");
            printStr += error;
            VCallCommand(printStr.c_str());
        }

        return true;
    }

    void CommandInterpreter::VLoadCommands(LPCSTR file)
    {        
        std::ifstream stream(file);
        if(stream.fail())
        {
            return;
        }

        std::string command;
        while(stream.good())
        {
            std::getline(stream, command);
            if(command.size() == 0 || (command[0] == '['))
            {
                continue;
            }
            VCallCommand(command.c_str());
        }
        stream.close();
    }

    CommandInterpreter::~CommandInterpreter(void)
    {

    }

    void TranslateActor(ActorId id, const util::Vec3& dTranslation)
    {
        TransformActor(id, dTranslation, util::Vec3());
    }

    void RotateActor(ActorId id, const util::Vec3& dRotation)
    {
        TransformActor(id, util::Vec3(), dRotation);
    }

    void TransformActor(ActorId id, const util::Vec3& dPostition, const util::Vec3& dRrotation)
    {
        QUEUE_EVENT(new MoveActorEvent(id, dPostition, dRrotation, true));
    }

    void SetActorPosition(ActorId id, const util::Vec3& position)
    {
        SetActorTransformation(id, position, util::Vec3());
    }

    void SetActorRotation(ActorId id, const util::Vec3& rotation)
    {
        SetActorTransformation(id, util::Vec3(), rotation);
    }

    void SetActorTransformation(ActorId id, const util::Vec3& postition, const util::Vec3& rotation)
    {
        QUEUE_EVENT(new chimera::MoveActorEvent(id, postition, rotation, false));
    }

    //--commands


    namespace commands
    {
        LPCSTR KEY_DOWN_STR = "down";
        LPCSTR KEY_RELEASED_STR = "released";
        LPCSTR KEY_PRESSEN_STR = "pressed";

        std::string CleanCommand(std::vector<std::string>& v)
        {
            std::string cmd;
            for(int i = 1; i < v.size(); ++i)
            {
                cmd += v[i] + " ";
            }
            return cmd;
        }

        bool Bind(ICommand& cmd)
        {
            std::string keyStr = cmd.VGetNextCharStr();
            CHECK_COMMAND(cmd);
            char key;
            if(keyStr.size() <= 1)
            {
                key = keyStr[0];
            }
            else
            {
                char* ptr = NULL;

                key = (char)strtol(keyStr.c_str(), &ptr, 16);

                if(ptr && ptr == keyStr.c_str())
                {
                    return false;
                }
            }
            
            std::string command = cmd.VGetRemainingString();
            CHECK_COMMAND(cmd);

            std::vector<std::string> split;
            util::split(command, ' ', split);
            int vk = GetVKFromchar(key);
            
            IActorController* controller = (IActorController*)(CmGetApp()->VGetLogic()->VFindView(VIEW_CONTROLLER_NAME));

            if(split.size() > 0 )
            {
                if(!strcmp(split[0].c_str(), KEY_DOWN_STR))
                {
                    controller->VRegisterKeyDownCommand(vk, CleanCommand(split));
                }
                else if(!strcmp(split[0].c_str(), KEY_RELEASED_STR))
                {
                    controller->VRegisterKeyReleasedCommand(vk, CleanCommand(split));
                }
                else if(!strcmp(split[0].c_str(), KEY_PRESSEN_STR))
                {
                    controller->VRegisterKeyPressedCommand(vk, CleanCommand(split));
                }
                else
                {
                    controller->VRegisterKeyCommand(vk, command);
                }
            }
            else
            {
                return false;
            }
            
            return true;
        }

        bool PlaySound(ICommand& cmd)
        {
            /*chimera::CMResource r(cmd.GetNextCharStr());
            if(!chimera::g_pApp->GetCache()->HasResource(r))
            {
                return FALSE;
            }
            std::shared_ptr<chimera::ResHandle> handle = chimera::g_pApp->GetCache()->GetHandle(r);
            std::shared_ptr<chimera::SoundProcess> proc = std::shared_ptr<chimera::SoundProcess>(new SoundProcess(handle));
            CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(proc);*/
            return true;
        }

        bool ToogleConsole(ICommand& cmd)
        {
            IScreenElement* cons = CmGetApp()->VGetHumanView()->VGetScreenElementByName(VIEW_CONSOLE_NAME);
            if(cons)
            {
                cons->VSetActive(!cons->VIsActive());
            }
            else
            {
                throw "No Console installed";
            }
            return true;
        }

        bool Print(ICommand& cmd)
        {
            GuiConsole* console = dynamic_cast<GuiConsole*>(CmGetApp()->VGetHumanView()->VGetScreenElementByName(VIEW_CONSOLE_NAME));
            if(console)
            {
                console->AppendText(cmd.VGetRemainingString());
            }
            return true;
        }

        bool SetTarget(LPCSTR actor)
        {
            IActor* a = CmGetApp()->VGetLogic()->VFindActor(actor);
            if(a)
            {
                CmGetApp()->VGetHumanView()->VSetTarget(a);
                CmGetApp()->VGetLogic()->VFindView("GameController")->VSetTarget(a);
            }
            return a != NULL;
        }

        bool End(ICommand& cmd)
        {
            CmGetApp()->VStopRunning();
            return true;
        }

        bool SetTarget(ICommand& cmd)
        {
            std::string actor = cmd.VGetNextCharStr();
            return SetTarget(actor.c_str());
        }

        bool ReloadLevel(ICommand& cmd)
        {
            return CmGetApp()->VGetLogic()->VLoadLevel(CmGetApp()->VGetLogic()->VGetlevel());
        }

        bool RunScript(ICommand& cmd)
        {
            std::string scriptFile = cmd.VGetNextCharStr();
            CHECK_COMMAND(cmd);
            if(!CmGetApp()->VGetScript())
            {
                throw "No Scripting available!";
            }
            CmGetApp()->VGetScript()->VRunFile((CmGetApp()->VGetConfig()->VGetString("sScriptPath") + scriptFile).c_str());
            return true;
        }

        bool SetSunPosition(ICommand& cmd)
        {
            float x = cmd.VGetNextFloat();
            float y = cmd.VGetNextFloat();
            float z = cmd.VGetNextFloat();
            CHECK_COMMAND(cmd);
            //event::IEventPtr event = std::shared_ptr<event::SetSunPositionEvent>();
            QUEUE_EVENT(new SetSunPositionEvent(x, y, z));
            return true;
        }

        bool SaveLevel(ICommand& cmd)
        {
            std::string name = cmd.VGetNextCharStr();
            CHECK_COMMAND(cmd);
            CmGetApp()->VGetLogic()->VGetlevel()->VSave(name.c_str());
            return true;
        }

        bool LoadLevel(ICommand& cmd)
        {
            std::string name = cmd.VGetNextCharStr();
            CHECK_COMMAND(cmd);

            std::string s;
            s += CmGetApp()->VGetCache()->VGetFile().VGetName();
            s += "/";
            s += CmGetApp()->VGetConfig()->VGetString("sLevelPath");
            s += name;

            if(!util::CheckIfFileExists(s.c_str()))
            {
                throw "File does not exist!";
            }

            chimera::LoadLevelEvent* e = new chimera::LoadLevelEvent();
            e->m_name = name;
            QUEUE_EVENT(e);

            return true;
        }

        bool RunProc(chimera::ICommand& cmd)
        {
            std::string cmdStr = cmd.VGetRemainingString();

            CHECK_COMMAND(cmd);

            STARTUPINFO si;
            ZeroMemory(&si, sizeof(si));
            AllocConsole();

            si.cb = sizeof(si);

            PROCESS_INFORMATION pi;
            ZeroMemory(&pi, sizeof(pi));

            SECURITY_ATTRIBUTES sa;
            ZeroMemory(&sa, sizeof(sa));
            sa.nLength = sizeof(sa);
            sa.lpSecurityDescriptor = NULL;
            sa.bInheritHandle = true;

            std::wstring ws(cmdStr.begin(), cmdStr.end());
            LPTSTR szCmdline = _tcsdup(ws.c_str());

            if(!CreateProcess(NULL, szCmdline, &sa, NULL, false, 0, NULL, NULL, &si, &pi))
            {
                return false;
            }

            WaitForSingleObject(pi.hProcess, INFINITE);

            DWORD exitCode;
            GetExitCodeProcess(pi.hProcess, &exitCode);

            FreeConsole();

            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);

            free(szCmdline);

            return exitCode > 0;
        }

        bool SpawnActor(chimera::ICommand& cmd)
        {
            std::string file = cmd.VGetNextCharStr();

            CHECK_COMMAND(cmd);

            if(!CmGetApp()->VGetCache()->VHasResource(CmGetApp()->VGetConfig()->VGetString("sActorPath") + chimera::CMResource(file)))
            {
                throw "File not found";
            }

            IActor* actor = CmGetApp()->VGetLogic()->VCreateActor(file.c_str());

            RETURN_IF_FAILED(actor);

            float x,y,z;

            x = cmd.VGetNextFloat();
            y = cmd.VGetNextFloat();
            z = cmd.VGetNextFloat();

            chimera::TransformComponent* cmp;
            chimera::CameraComponent* camCmp;

            if(cmd.VIsError())
            {
                CmGetApp()->VGetHumanView()->VGetTarget()->VQueryComponent(CM_CMP_TRANSFORM, (chimera::IActorComponent**)&cmp);
                CmGetApp()->VGetHumanView()->VGetTarget()->VQueryComponent(CM_CMP_CAMERA, (chimera::IActorComponent**)&camCmp);
                x = cmp->GetTransformation()->GetTranslation().x + 6*camCmp->GetCamera()->GetViewDir().x;
                y = cmp->GetTransformation()->GetTranslation().y + 6*camCmp->GetCamera()->GetViewDir().y;
                z = cmp->GetTransformation()->GetTranslation().z + 6*camCmp->GetCamera()->GetViewDir().z;
            }

            actor->VQueryComponent(CM_CMP_TRANSFORM, (chimera::IActorComponent**)&cmp);
            cmp->GetTransformation()->SetTranslation(util::Vec3(x, y, z));

            return true;
        }

        bool Jump(chimera::ICommand& cmd)
        {
            float height = cmd.VGetNextFloat();
            CHECK_COMMAND(cmd);
            ActorId id = CmGetApp()->VGetHumanView()->VGetTarget()->GetId();
            chimera::MoveActorEvent* me = new chimera::MoveActorEvent(id, util::Vec3(0, height, 0));
            me->m_isJump = true;
            QUEUE_EVENT(me);
            return true;
        }

        bool SetRenderMode(chimera::ICommand& cmd)
        {
            std::string mode = cmd.VGetNextCharStr();

            if(cmd.VIsError())
            {
                return false;
            }

            CmGetApp()->VGetHumanView()->VActivateScene(mode.c_str());
            return true;
        }

        ActorId lastActor = CM_INVALID_ACTOR_ID;
        IActorComponent* tmpCmp = NULL;
        void callBack(ActorId id)
        {
            IActor* playerActor = CmGetApp()->VGetHumanView()->VGetTarget();

            CameraComponent* cameraCmp;
            playerActor->VQueryComponent(CM_CMP_CAMERA, (IActorComponent**)&cameraCmp);

            util::Vec3 eyePos;
            util::Vec3 offset(0, 0, 5);
            if(cameraCmp)
            {
                eyePos = cameraCmp->GetCamera()->GetEyePos();
                offset.y = cameraCmp->GetCamera()->GetYOffset();
            }

            if(lastActor != CM_INVALID_ACTOR_ID)
            {
                ReleaseChildEvent* pe = new ReleaseChildEvent();
                pe->m_actor = lastActor;
                QUEUE_EVENT(pe);

                TransformComponent* tCmp;
                IActor* targetActor = CmGetApp()->VGetLogic()->VFindActor(lastActor);
                TRIGGER_EVENT(new MoveActorEvent(lastActor, (cameraCmp->GetCamera()->GetViewDir() * 5) + eyePos, false));

                if(tmpCmp)
                {
                    targetActor->VAddComponent(std::unique_ptr<IActorComponent>(tmpCmp));
                    QUEUE_EVENT(new NewComponentCreatedEvent(tmpCmp->VGetComponentId(), targetActor->GetId()));
                }

                lastActor = CM_INVALID_ACTOR_ID;
                return;
            }

            if(id != CM_INVALID_ACTOR_ID && id != lastActor)
            {
                SetParentActorEvent* pe = new SetParentActorEvent();
                pe->m_actor = id;
                pe->m_parentActor = CmGetApp()->VGetHumanView()->VGetTarget()->GetId();
                QUEUE_EVENT(pe);
                IActor* targetActor = CmGetApp()->VGetLogic()->VFindActor(id);

                TransformComponent* tCmp;
                targetActor->VQueryComponent(CM_CMP_TRANSFORM, (IActorComponent**)&tCmp);
                tCmp->GetTransformation()->SetTranslation(offset);
                QUEUE_EVENT(new ActorMovedEvent(CmGetApp()->VGetLogic()->VFindActor(id)));

                tmpCmp = targetActor->VReleaseComponent(CM_CMP_PHX).release();
                if(tmpCmp)
                {
                    QUEUE_EVENT(new DeleteComponentEvent(targetActor, tmpCmp));
                }
            }

            lastActor = id;
        }

        bool PickActor(chimera::ICommand& cmd)
        {
            IEventPtr event(new PickActorEvent(callBack));
            TRIGGER_EVENT(event);
            return true;
        }

        bool DeleteSelectedActor(chimera::ICommand& cmd)
        {
            ActorId id = lastActor;
            lastActor = CM_INVALID_ACTOR_ID;
            if(id == CM_INVALID_ACTOR_ID)
            {
                return false;
            }

            DeleteActorEvent* e = new DeleteActorEvent(id);
            QUEUE_EVENT(e);
            return true;
        }

        bool DeleteActor(chimera::ICommand& cmd)
        {
            ActorId id = (ActorId)cmd.VGetNextInt();
            CHECK_COMMAND(cmd);
            DeleteActorEvent* e = new DeleteActorEvent(id);
            QUEUE_EVENT(e);
            return true;
        }

        bool ApplyPlayerForce(ICommand& cmd)
        {
            IActor* playerActor = CmGetApp()->VGetHumanView()->VGetTarget();

            CameraComponent* cameraCmp;
            playerActor->VQueryComponent(CM_CMP_CAMERA, (IActorComponent**)&cameraCmp);

            const util::Vec3& dir = cameraCmp->GetCamera()->GetViewDir();

            if(lastActor != CM_INVALID_ACTOR_ID)
            {
                ReleaseChildEvent* pe = new ReleaseChildEvent();
                pe->m_actor = lastActor;
                TRIGGER_EVENT(pe);

                TransformComponent* tCmp;
                IActor* targetActor = CmGetApp()->VGetLogic()->VFindActor(lastActor);
                TRIGGER_EVENT(new MoveActorEvent(lastActor, (cameraCmp->GetCamera()->GetViewDir() * 5) + cameraCmp->GetCamera()->GetEyePos(), false));

                if(tmpCmp)
                {
                    targetActor->VAddComponent(std::unique_ptr<IActorComponent>(tmpCmp));
                    TRIGGER_EVENT(new NewComponentCreatedEvent(tmpCmp->VGetComponentId(), targetActor->GetId()));
                }
            }

            util::Vec3 tdir = dir;
            tdir.Scale(0.25f);
            std::stringstream ss;
            ss << "force ";
            ss << tdir.x << " ";
            ss << tdir.y << " ";
            ss << tdir.z;
            CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand(ss.str().c_str());

            ss.str("");
            ss << "torque ";
            ss << dir.x << " ";
            ss << dir.y << " ";
            ss << dir.z;
            CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand(ss.str().c_str());

            return true;
        }

        bool ApplyForce(ICommand& cmd)
        {
            if(lastActor == CM_INVALID_ACTOR_ID)
            {
                return true;
            }

            float x = cmd.VGetNextFloat();
            float y = cmd.VGetNextFloat();
            float z = cmd.VGetNextFloat();
            CHECK_COMMAND(cmd);

            float n = cmd.VGetNextFloat();

            n = n == 0 ? 1 : n;

            util::Vec3 dir(x, y, z);

            ApplyForceEvent* ev = new ApplyForceEvent();
            ev->m_actor = CmGetApp()->VGetLogic()->VFindActor(lastActor);
            ev->m_dir = dir;
            ev->m_newtons = 100000;

            QUEUE_EVENT(ev);

            lastActor = CM_INVALID_ACTOR_ID;

            return true;
        }

        bool ApplyTorque(ICommand& cmd)
        {
            if(lastActor == CM_INVALID_ACTOR_ID)
            {
                return true;
            }

            float x = cmd.VGetNextFloat();
            float y = cmd.VGetNextFloat();
            float z = cmd.VGetNextFloat();
            CHECK_COMMAND(cmd);

            float n = cmd.VGetNextFloat();

            n = n == 0 ? 1 : n;

            util::Vec3 dir(x, y, z);

            ApplyTorqueEvent* ev = new ApplyTorqueEvent();
            ev->m_actor = CmGetApp()->VGetLogic()->VFindActor(lastActor);
            ev->m_torque = dir;
            ev->m_newtons = 100000;

            QUEUE_EVENT(ev);

            //m_toModify = CM_INVALID_ACTOR_ID;
            lastActor = CM_INVALID_ACTOR_ID;

            //m_bMovePicked = false;

            return true;
        }

        util::Vec3 sunIntensity(1,1,1);
        bool IncSunIntensity(ICommand& cmd)
        {
            sunIntensity = sunIntensity + util::Vec3(0.1f, 0.1f, 0.1f);
            SetSunIntensityEvent* e = new SetSunIntensityEvent(sunIntensity.x, sunIntensity.y, sunIntensity.z);
            QUEUE_EVENT(e);
            return true;
        }

        bool DecSunIntensity(ICommand& cmd)
        {
            sunIntensity = sunIntensity - util::Vec3(0.1f, 0.1f, 0.1f);
            SetSunIntensityEvent* e = new SetSunIntensityEvent(sunIntensity.x, sunIntensity.y, sunIntensity.z);
            QUEUE_EVENT(e);
            return true;
        }

        util::Vec3 sunAmbient(0.1f,0.1f,0.1f);
        bool IncSunAmbient(ICommand& cmd)
        {
            sunAmbient = sunAmbient + util::Vec3(0.1f, 0.1f, 0.1f);
            SetSunAmbient* e = new SetSunAmbient(sunAmbient.x, sunAmbient.y, sunAmbient.z);
            QUEUE_EVENT(e);
            return true;
        }

        bool DecSunAmbient(ICommand& cmd)
        {
            sunAmbient = sunAmbient - util::Vec3(0.1f, 0.1f, 0.1f);
            SetSunAmbient* e = new SetSunAmbient(sunAmbient.x, sunAmbient.y, sunAmbient.z);
            QUEUE_EVENT(e);
            return true;
        }

        void AddCommandsToInterpreter(CommandInterpreter& interpreter)
        {
            interpreter.VRegisterCommand("runproc", RunProc);
            interpreter.VRegisterCommand("bind", commands::Bind, "bind [key] [command]");
            interpreter.VRegisterCommand("playsound", commands::PlaySound, "playsound [valid wavefile]");
            interpreter.VRegisterCommand("console", commands::ToogleConsole);
            interpreter.VRegisterCommand("print", commands::Print, "print [some string]");
            interpreter.VRegisterCommand("target", commands::SetTarget, "target [some actorname]");
            interpreter.VRegisterCommand("reload", commands::ReloadLevel);
            interpreter.VRegisterCommand("runscript", commands::RunScript, "runscript [input file]");
            interpreter.VRegisterCommand("sunpos", commands::SetSunPosition, "sunpos x y z");
            interpreter.VRegisterCommand("savelevel", commands::SaveLevel, "savelevel [levelname]");
            interpreter.VRegisterCommand("loadlevel", commands::LoadLevel, "loadLevel [levelname]");
            interpreter.VRegisterCommand("spawnactor", commands::SpawnActor, "spawnactor [filename]");
            interpreter.VRegisterCommand("deleteactor", commands::DeleteActor, "deleteactor [id]");
            interpreter.VRegisterCommand("deleteselectedactor", commands::DeleteSelectedActor, "deleteselectedactor");
            interpreter.VRegisterCommand("r_setmode", commands::SetRenderMode, "r_setmode [mode]");
            interpreter.VRegisterCommand("jump", commands::Jump, "jump");
            interpreter.VRegisterCommand("pick", commands::PickActor, "pick");
            interpreter.VRegisterCommand("pforce", commands::ApplyPlayerForce, "pforce");
            interpreter.VRegisterCommand("force", commands::ApplyForce, "force [id x y z]");
            interpreter.VRegisterCommand("inc_sun_i", commands::IncSunIntensity);
            interpreter.VRegisterCommand("dec_sun_i", commands::DecSunIntensity);
            interpreter.VRegisterCommand("inc_sun_a", commands::IncSunAmbient);
            interpreter.VRegisterCommand("dec_sun_a", commands::DecSunAmbient);
            interpreter.VRegisterCommand("torque", commands::ApplyTorque, "torque [id x y z]");
            interpreter.VRegisterCommand("exit", commands::End);
        }
    }
}
