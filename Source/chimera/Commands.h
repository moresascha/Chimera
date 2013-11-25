#pragma once
#include "stdafx.h"
#include "Vec3.h"
#include "Actor.h"
#include "Config.h"
#include <vector>

#define CHECK_COMMAND(__cmd) if(__cmd.VIsError()) { return FALSE; }

namespace chimera
{
    class Command : public ICommand
    {
    private:
        std::list<INT> m_argList;
        std::list<std::string> m_values;
        BOOL m_error;
    public:
        Command(std::list<std::string>& elems);
        
        BOOL VInitArgumentTypes(INT args, ...);
        
        FLOAT VGetNextFloat(VOID);
        INT VGetNextInt(VOID);
        CHAR VGetNextChar(VOID);
        BOOL VGetNextBool(VOID);
        std::string VGetNextCharStr(VOID);
        std::string VGetRemainingString(VOID);
        BOOL VIsError(VOID);
        BOOL VIsValid(VOID);

        ~Command(VOID);
    };

    class CommandInterpreter : public ICommandInterpreter
    {
    private:
        std::map<std::string, CommandHandler> m_nameToCommandHandler;

        std::map<std::string, std::string> m_nameToUsage;

    public:
        CommandInterpreter(VOID);

        VOID VRegisterCommand(LPCSTR name, CommandHandler command, LPCSTR usage = NULL);

        BOOL VCallCommand(LPCSTR command);

        std::vector<std::string> VGetCommands(VOID);

        VOID VLoadCommands(LPCSTR file);

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
        BOOL Bind(ICommand& cmd);

        BOOL PlaySound(ICommand& cmd);

        BOOL ToogleConsole(ICommand& cmd);
        
        BOOL SpawnMeshActor(ICommand& cmd);

        BOOL SetTarget(ICommand& cmd);

        BOOL SetTarget(LPCSTR actor);

        BOOL Print(ICommand& cmd);

        BOOL End(ICommand& cmd);

        VOID AddCommandsToInterpreter(CommandInterpreter& interpreter);
    }
}

