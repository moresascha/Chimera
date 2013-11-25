#include "Commands.h"
#include "util.h"
#include <fstream>
#include "Event.h"
namespace chimera
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

    BOOL Command::VInitArgumentTypes(INT args, ...)
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

    std::string Command::VGetRemainingString(VOID)
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

    FLOAT Command::VGetNextFloat(VOID)
    {
        CHECK_FOR_ERROR((FLOAT)0.0f);

        CHAR* ptr = NULL;
        CONST CHAR* toConvert = m_values.begin()->c_str();

        FLOAT v = (FLOAT)strtod(toConvert, &ptr);

        m_error = ptr && ptr == toConvert;

        m_values.pop_front();
        return v;
    }

    BOOL Command::VGetNextBool(VOID)
    {
        m_error = CheckForError(m_values);
        return VGetNextInt() != 0;
    }

    INT Command::VGetNextInt(VOID)
    {
        CHECK_FOR_ERROR((INT)0);

        CHAR* ptr = NULL;
        CONST CHAR* toConvert = m_values.begin()->c_str();

        INT v = (INT)strtol(toConvert, &ptr, 10);

        m_error = ptr && ptr == toConvert;

        m_values.pop_front();
        return v;
    }

    CHAR Command::VGetNextChar(VOID)
    {
        CHECK_FOR_ERROR((CHAR)0);

        std::string s = *m_values.begin();
        m_values.pop_front();
        return (CHAR)s.c_str()[0];
    }

    std::string Command::VGetNextCharStr(VOID)
    {
        CHECK_FOR_ERROR(std::string());

        std::string s = *m_values.begin();
        m_values.pop_front();
        return s;
    }

    BOOL Command::VIsError(VOID)
    {
        return m_error;
    }

    BOOL Command::VIsValid(VOID)
    {
        return !VIsError();
    }

    Command::~Command(VOID)
    {

    }

    CommandInterpreter::CommandInterpreter(VOID)
    {
        commands::AddCommandsToInterpreter(*this);
    }

    VOID CommandInterpreter::VRegisterCommand(LPCSTR name, CommandHandler command, LPCSTR usage)
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

    std::vector<std::string> CommandInterpreter::VGetCommands(VOID)
    {
        std::vector<std::string> list;
        TBD_FOR(m_nameToCommandHandler)
        {
            list.push_back(it->first);
        }
        return list;
    }

    BOOL CommandInterpreter::VCallCommand(LPCSTR command)
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

        return TRUE;
    }

    VOID CommandInterpreter::VLoadCommands(LPCSTR file)
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
        QUEUE_EVENT(new MoveActorEvent(id, dPostition, dRrotation, TRUE));
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
        QUEUE_EVENT(new chimera::MoveActorEvent(id, postition, rotation, FALSE));
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

        BOOL Bind(ICommand& cmd)
        {
            std::string keyStr = cmd.VGetNextCharStr();
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
            
            std::string command = cmd.VGetRemainingString();
            CHECK_COMMAND(cmd);

            std::vector<std::string> split;
            util::split(command, ' ', split);
            INT vk = GetVKFromchar(key);
            
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
                return FALSE;
            }
            
            return TRUE;
        }

        BOOL PlaySound(ICommand& cmd)
        {
            /*chimera::CMResource r(cmd.GetNextCharStr());
            if(!chimera::g_pApp->GetCache()->HasResource(r))
            {
                return FALSE;
            }
            std::shared_ptr<chimera::ResHandle> handle = chimera::g_pApp->GetCache()->GetHandle(r);
            std::shared_ptr<chimera::SoundProcess> proc = std::shared_ptr<chimera::SoundProcess>(new SoundProcess(handle));
            CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(proc);*/
            return TRUE;
        }

        BOOL ToogleConsole(ICommand& cmd)
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
            return TRUE;
        }

        BOOL Print(ICommand& cmd)
        {
            CmGetApp()->VGetHumanView()->VGetScreenElementByName(VIEW_CONSOLE_NAME);
            return TRUE;
        }

        BOOL SetTarget(LPCSTR actor)
        {
            IActor* a = CmGetApp()->VGetLogic()->VFindActor(actor);
            if(a)
            {
                CmGetApp()->VGetHumanView()->VSetTarget(a);
                CmGetApp()->VGetLogic()->VFindView("GameController")->VSetTarget(a);
            }
            return a != NULL;
        }

        BOOL End(ICommand& cmd)
        {
            CmGetApp()->VStopRunning();
            return TRUE;
        }

        BOOL SetTarget(ICommand& cmd)
        {
            std::string actor = cmd.VGetNextCharStr();
            return SetTarget(actor.c_str());
        }

        BOOL ReloadLevel(ICommand& cmd)
        {
            return CmGetApp()->VGetLogic()->VLoadLevel(CmGetApp()->VGetLogic()->VGetlevel());
        }

        BOOL RunScript(ICommand& cmd)
        {
            std::string scriptFile = cmd.VGetNextCharStr();
            CHECK_COMMAND(cmd);
            if(!CmGetApp()->VGetScript())
            {
                throw "No Scripting available!";
            }
            CmGetApp()->VGetScript()->VRunFile((CmGetApp()->VGetConfig()->VGetString("sScriptPath") + scriptFile).c_str());
            return TRUE;
        }

        BOOL SetSunPosition(ICommand& cmd)
        {
            FLOAT x = cmd.VGetNextFloat();
            FLOAT y = cmd.VGetNextFloat();
            FLOAT z = cmd.VGetNextFloat();
            CHECK_COMMAND(cmd);
            //event::IEventPtr event = std::shared_ptr<event::SetSunPositionEvent>();
            QUEUE_EVENT(new SetSunPositionEvent(x, y, z));
            return TRUE;
        }

        BOOL SaveLevel(ICommand& cmd)
        {
            std::string name = cmd.VGetNextCharStr();
            CHECK_COMMAND(cmd);
            CmGetApp()->VGetLogic()->VGetlevel()->VSave(name.c_str());
            return TRUE;
        }

        BOOL LoadLevel(ICommand& cmd)
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

            return TRUE;
        }

        BOOL RunProc(chimera::ICommand& cmd)
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
            interpreter.VRegisterCommand("loadlevel", commands::LoadLevel, "LoadLevel [levelname]");
            interpreter.VRegisterCommand("exit", commands::End);
        }
    }
}
