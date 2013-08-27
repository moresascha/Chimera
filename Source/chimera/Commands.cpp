#include "Commands.h"
#include "GameApp.h"
#include "util.h"
#include <fstream>
#include "GameView.h"
#include "EventManager.h"
#include "GameLogic.h"
#include "Resources.h"
#include "Process.h"
#include "ProcessManager.h"
#include <errno.h>
#include "GuiComponent.h"
#include "Components.h"
#include "Script.h"
namespace tbd
{

    BOOL CheckForError(std::list<std::string> toCheck)
    {
        return toCheck.size() == 0;
    }

#define CHECK_FOR_ERROR(_RET) \
    if(CheckForError(m_values)) \
    { \
    m_error = TRUE; \
    return _RET; \
    }

    Command::Command(std::list<std::string>& elems) : m_values(elems), m_error(FALSE)
    {
    }

    BOOL Command::InitArgumentTypes(INT args, ...)
    {
        va_list pointer;

        va_start(pointer, args);

        for(INT i = 0; i < args; ++i)
        {
            INT format = va_arg(pointer, INT);
            m_argList.push_back(format);
        }

        va_end(pointer);

        return TRUE;
    }

    std::string Command::GetRemainingString(VOID)
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

    FLOAT Command::GetNextFloat(VOID)
    {
        CHECK_FOR_ERROR((FLOAT)0.0f);

        CHAR* ptr = NULL;
        CONST CHAR* toConvert = m_values.begin()->c_str();

        FLOAT v = (FLOAT)strtod(toConvert, &ptr);

        m_error = ptr && ptr == toConvert;

        m_values.pop_front();
        return v;
    }

    BOOL Command::GetNextBool(VOID)
    {
        m_error = CheckForError(m_values);
        return GetNextInt() != 0;
    }

    INT Command::GetNextInt(VOID)
    {
        CHECK_FOR_ERROR((INT)0);

        CHAR* ptr = NULL;
        CONST CHAR* toConvert = m_values.begin()->c_str();

        INT v = (INT)strtol(toConvert, &ptr, 10);

        m_error = ptr && ptr == toConvert;

        m_values.pop_front();
        return v;
    }

    CHAR Command::GetNextChar(VOID)
    {
        CHECK_FOR_ERROR((CHAR)0);

        std::string s = *m_values.begin();
        m_values.pop_front();
        return (CHAR)s.c_str()[0];
    }

    std::string Command::GetNextCharStr(VOID)
    {
        CHECK_FOR_ERROR(std::string());

        std::string s = *m_values.begin();
        m_values.pop_front();
        return s;
    }

    BOOL Command::IsError(VOID)
    {
        return m_error;
    }

    BOOL Command::IsValid(VOID)
    {
        return !IsError();
    }

    Command::~Command(VOID)
    {

    }

    CommandInterpreter::CommandInterpreter(VOID)
    {
    }

    VOID CommandInterpreter::RegisterCommand(LPCSTR name, CommandHandler command, LPCSTR usage)
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

    std::list<std::string> CommandInterpreter::GetCommands(VOID)
    {
        std::list<std::string> list;
        TBD_FOR(m_nameToCommandHandler)
        {
            list.push_back(it->first);
        }
        return list;
    }

    BOOL CommandInterpreter::CallCommand(LPCSTR command)
    {
        std::vector<std::string> elements = util::split(std::string(command), ' ');
        if(elements.size() == 0)
        {
            return FALSE;
        }
        //CHECK_VETOR_SIZE(elements);

        std::string& cmd = elements.front();

        auto it = m_nameToCommandHandler.find(std::string(cmd));

        if(it == m_nameToCommandHandler.end())
        {
            //DEBUG_OUT("no commandhandler for command: " + cmd);
            return FALSE;
        }

        std::list<std::string> vals;

        for(INT i = 1; i < elements.size(); ++i)
        {
            vals.push_back(elements[i]);
        }

        tbd::Command c(vals);
        try 
        {
            if(!it->second(c))
            {
                std::string printStr("print " + m_nameToUsage[cmd]);
                CallCommand(printStr.c_str());
            }
        } catch(LPCSTR error)
        {
            std::string printStr("print ");
            printStr += error;
            CallCommand(printStr.c_str());
        }


        return TRUE;
    }

    /*static*/ BOOL CommandInterpreter::LoadCommands(LPCSTR file, CommandInterpreter& interpreter)
    {
        std::ifstream stream(file);
        if(stream.fail())
        {
            return FALSE;
        }

        std::string command;
        while(stream.good())
        {
            std::getline(stream, command);
            if(command.size() == 0 || (command[0] == '['))
            {
                continue;
            }
            interpreter.CallCommand(command.c_str());
        }
        stream.close();

        return TRUE;
    }

    CommandInterpreter::~CommandInterpreter(VOID)
    {

    }

    VOID TranslateActor(ActorId id, CONST util::Vec3& dTranslation)
    {
        TransformActor(id, dTranslation, util::Vec3());
    }

    VOID RotateActor(ActorId id, CONST util::Vec3& dRotation)
    {
        TransformActor(id, util::Vec3(), dRotation);
    }

    VOID TransformActor(ActorId id, CONST util::Vec3& dPostition, CONST util::Vec3& dRrotation)
    {
        event::IEventPtr event(new event::MoveActorEvent(id, dPostition, dRrotation, TRUE));
        event::IEventManager::Get()->VQueueEvent(event); 
    }

    VOID SetActorPosition(ActorId id, CONST util::Vec3& position)
    {
        SetActorTransformation(id, position, util::Vec3());
    }

    VOID SetActorRotation(ActorId id, CONST util::Vec3& rotation)
    {
        SetActorTransformation(id, util::Vec3(), rotation);
    }

    VOID SetActorTransformation(ActorId id, CONST util::Vec3& postition, CONST util::Vec3& rotation)
    {
        event::IEventPtr event(new event::MoveActorEvent(id, postition, rotation, FALSE));
        event::IEventManager::Get()->VQueueEvent(event); 
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

        BOOL Bind(Command& cmd)
        {
            std::string keyStr = cmd.GetNextCharStr();
            CHECK_COMMAND(cmd);
            CHAR key;
            if(keyStr.size() <= 1)
            {
                key = keyStr[0];
            }
            else
            {
                CHAR* ptr = NULL;

                key = (CHAR)strtol(keyStr.c_str(), &ptr, 16);

                if(ptr && ptr == keyStr.c_str())
                {
                    return FALSE;
                }
            }
            
            std::string command = cmd.GetRemainingString();
            CHECK_COMMAND(cmd);

            std::vector<std::string> split;
            util::split(command, ' ', split);
            INT vk = GetVKFromchar(key);
            
            std::shared_ptr<tbd::ActorController> controller = std::static_pointer_cast<tbd::ActorController>(app::g_pApp->GetLogic()->VFindGameView("GameController"));

            if(split.size() > 0 )
            {
                if(!strcmp(split[0].c_str(), KEY_DOWN_STR))
                {
                    controller->RegisterKeyDownCommand(vk, CleanCommand(split));
                }
                else if(!strcmp(split[0].c_str(), KEY_RELEASED_STR))
                {
                    controller->RegisterKeyReleasedCommand(vk, CleanCommand(split));
                }
                else if(!strcmp(split[0].c_str(), KEY_PRESSEN_STR))
                {
                    controller->RegisterKeyPressedCommand(vk, CleanCommand(split));
                }
                else
                {
                    controller->RegisterKeyCommand(vk, command);
                }
            }
            else
            {
                return FALSE;
            }
            
            return TRUE;
        }

        BOOL PlaySound(Command& cmd)
        {
            tbd::Resource r(cmd.GetNextCharStr());
            if(!app::g_pApp->GetCache()->HasResource(r))
            {
                return FALSE;
            }
            std::shared_ptr<tbd::ResHandle> handle = app::g_pApp->GetCache()->GetHandle(r);
            std::shared_ptr<proc::SoundProcess> proc = std::shared_ptr<proc::SoundProcess>(new proc::SoundProcess(handle));
            app::g_pApp->GetLogic()->AttachProcess(proc);
            return TRUE;
        }

        BOOL ToogleConsole(Command& cmd)
        {
            app::g_pApp->GetHumanView()->ToggleConsole();
            return TRUE;
        }

        BOOL Print(Command& cmd)
        {
            app::g_pApp->GetHumanView()->GetConsole()->AppendText(cmd.GetRemainingString());
            return TRUE;
        }

        BOOL SetTarget(LPCSTR actor)
        {
            std::shared_ptr<tbd::Actor> a = app::g_pApp->GetLogic()->VFindActor(actor);
            if(a)
            {
                app::g_pApp->GetHumanView()->VSetTarget(a);
                app::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(a);
            }
            return a != NULL;
        }

        BOOL SetTarget(Command& cmd)
        {
            std::string actor = cmd.GetNextCharStr();
            return SetTarget(actor.c_str());
        }

        BOOL ReloadLevel(Command& cmd)
        {
            return app::g_pApp->GetLogic()->VLoadLevel(app::g_pApp->GetLogic()->Getlevel());
        }

        BOOL RunScript(Command& cmd)
        {
            std::string scriptFile = cmd.GetNextCharStr();
            CHECK_COMMAND(cmd);
            app::g_pApp->GetScript()->VRunFile((app::g_pApp->GetConfig()->GetString("sScriptPath") + scriptFile).c_str());
            return TRUE;
        }

        BOOL SetSunPosition(Command& cmd)
        {
            FLOAT x = cmd.GetNextFloat();
            FLOAT y = cmd.GetNextFloat();
            FLOAT z = cmd.GetNextFloat();
            CHECK_COMMAND(cmd);
            //event::IEventPtr event = std::shared_ptr<event::SetSunPositionEvent>();
            QUEUE_EVENT(new event::SetSunPositionEvent(x, y, z));
            return TRUE;
        }

        BOOL SaveLevel(Command& cmd)
        {
            std::string name = cmd.GetNextCharStr();
            CHECK_COMMAND(cmd);
            app::g_pApp->GetLogic()->Getlevel()->VSave(name.c_str());
            return TRUE;
        }

        BOOL LoadLevel(tbd::Command& cmd)
        {
            std::string name = cmd.GetNextCharStr();
            CHECK_COMMAND(cmd);

            std::string s;
            s += app::g_pApp->GetCache()->GetFile().VGetName();
            s += "/";
            s += app::g_pApp->GetConfig()->GetString("sLevelPath");
            s += name;

            if(!util::CheckIfFileExists(s.c_str()))
            {
                throw "File does not exist!";
            }

            event::LoadLevelEvent* e = new event::LoadLevelEvent();
            e->m_name = name;
            QUEUE_EVENT(e);

            return TRUE;
        }

        BOOL RunProc(tbd::Command& cmd)
        {
            std::string cmdStr = cmd.GetRemainingString();

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
            sa.bInheritHandle = TRUE;

            std::wstring ws(cmdStr.begin(), cmdStr.end());
            LPTSTR szCmdline = _tcsdup(ws.c_str());

            if(!CreateProcess(NULL, szCmdline, &sa, NULL, FALSE, 0, NULL, NULL, &si, &pi))
            {
                return FALSE;
            }

            WaitForSingleObject(pi.hProcess, INFINITE);

            DWORD exitCode;
            GetExitCodeProcess(pi.hProcess, &exitCode);

            FreeConsole();

            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);

            free(szCmdline);

            return exitCode;
        }

        VOID AddCommandsToInterpreter(CommandInterpreter& interpreter)
        {
            interpreter.RegisterCommand("runproc", RunProc);
            interpreter.RegisterCommand("bind", commands::Bind, "bind [key] [command]");
            interpreter.RegisterCommand("playsound", commands::PlaySound, "playsound [valid wavefile]");
            interpreter.RegisterCommand("console", commands::ToogleConsole);
            interpreter.RegisterCommand("print", commands::Print, "print [some string]");
            interpreter.RegisterCommand("target", commands::SetTarget, "target [some actorname]");
            interpreter.RegisterCommand("reload", commands::ReloadLevel);
            interpreter.RegisterCommand("runscript", commands::RunScript);
            interpreter.RegisterCommand("sunpos", commands::SetSunPosition, "sunpos x y z");
            interpreter.RegisterCommand("savelevel", commands::SaveLevel, "savelevel [levelname]");
            interpreter.RegisterCommand("loadlevel", commands::LoadLevel, "LoadLevel [levelname]");
        }
    }
}
