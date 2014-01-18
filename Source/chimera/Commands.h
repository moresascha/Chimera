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
        std::list<int> m_argList;
        std::list<std::string> m_values;
        bool m_error;
    public:
        Command(std::list<std::string>& elems);
        
        bool VInitArgumentTypes(int args, ...);
        
        float VGetNextFloat(void);
        int VGetNextInt(void);
        char VGetNextChar(void);
        bool VGetNextBool(void);
        std::string VGetNextCharStr(void);
        std::string VGetRemainingString(void);
        bool VIsError(void);
        bool VIsValid(void);

        ~Command(void);
    };

    class CommandInterpreter : public ICommandInterpreter
    {
    private:
        std::map<std::string, CommandHandler> m_nameToCommandHandler;

        std::map<std::string, std::string> m_nameToUsage;

    public:
        CommandInterpreter(void);

        void VRegisterCommand(LPCSTR name, CommandHandler command, LPCSTR usage = NULL);

        bool VCallCommand(LPCSTR command);

        std::vector<std::string> VGetCommands(void);

        void VLoadCommands(LPCSTR file);

        ~CommandInterpreter(void);
    };

    void TranslateActor(ActorId id, const util::Vec3& dTranslation);

    void RotateActor(ActorId id, const util::Vec3& dRotation);

    void TransformActor(ActorId id, const util::Vec3& dPostition, const util::Vec3& dRrotation);

    void SetActorPosition(ActorId id, const util::Vec3& position);

    void SetActorRotation(ActorId id, const util::Vec3& position);

    void SetActorTransformation(ActorId id, const util::Vec3& position, const util::Vec3& rotation);

    void SetRenderMode(int mode);

    //some usefull commands
    namespace commands
    {
        bool Bind(ICommand& cmd);

        bool PlaySound(ICommand& cmd);

        bool ToogleConsole(ICommand& cmd);
        
        bool SpawnMeshActor(ICommand& cmd);

        bool SetTarget(ICommand& cmd);

        bool SetTarget(LPCSTR actor);

        bool Print(ICommand& cmd);

        bool End(ICommand& cmd);

        void AddCommandsToInterpreter(CommandInterpreter& interpreter);
    }
}

