#pragma once
#include "stdafx.h"
#include "Vec3.h"
#include "Actor.h"
#include "Config.h"
#include <vector>

#define CHECK_COMMAND(__cmd) if(__cmd.IsError()) { return FALSE; }

namespace chimera
{
    class Command
    {
    private:
        std::list<INT> m_argList;
        std::list<std::string> m_values;
        BOOL m_error;
    public:
        Command(std::list<std::string>& elems);
        
        BOOL InitArgumentTypes(INT args, ...);
        
        CM_DLL_API FLOAT GetNextFloat(VOID);
        CM_DLL_API INT GetNextInt(VOID);
        CM_DLL_API CHAR GetNextChar(VOID);
        CM_DLL_API BOOL GetNextBool(VOID);
        CM_DLL_API std::string GetNextCharStr(VOID);
        CM_DLL_API std::string GetRemainingString(VOID);
        CM_DLL_API BOOL IsError(VOID);
        CM_DLL_API BOOL IsValid(VOID);

        ~Command(VOID);
    };

    class CommandInterpreter : public ICommandInterpreter
    {
    private:
        std::map<std::string, CommandHandler> m_nameToCommandHandler;

        std::map<std::string, std::string> m_nameToUsage;

    public:
        CommandInterpreter(VOID);

        VOID RegisterCommand(LPCSTR name, CommandHandler command, LPCSTR usage = NULL);

        BOOL VCallCommand(LPCSTR command);

        std::list<std::string> GetCommands(VOID);

        static BOOL LoadCommands(LPCSTR file, CommandInterpreter& interpreter);

        ~CommandInterpreter(VOID);
    };

    VOID TranslateActor(ActorId id, CONST util::Vec3& dTranslation);

    VOID RotateActor(ActorId id, CONST util::Vec3& dRotation);

    VOID TransformActor(ActorId id, CONST util::Vec3& dPostition, CONST util::Vec3& dRrotation);

    VOID SetActorPosition(ActorId id, CONST util::Vec3& position);

    VOID SetActorRotation(ActorId id, CONST util::Vec3& position);

    VOID SetActorTransformation(ActorId id, CONST util::Vec3& position, CONST util::Vec3& rotation);

    VOID SetRenderMode(int mode);

    //some usefull commands
    namespace commands
    {
        BOOL Bind(Command& cmd);

        BOOL PlaySound(Command& cmd);

        BOOL ToogleConsole(Command& cmd);
        
        BOOL SpawnMeshActor(Command& cmd);

        BOOL SetTarget(Command& cmd);

        BOOL SetTarget(LPCSTR actor);

        BOOL Print(Command& cmd);

        VOID AddCommandsToInterpreter(CommandInterpreter& interpreter);
    }
}

