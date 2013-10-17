#pragma once
#include "stdafx.h"
#include "GameView.h"
#include "Commands.h"

namespace gameinput
{
    VOID PlayTestSound();

    BOOL SetRenderMode(chimera::Command& cmd);

    BOOL SetDefaultPlayer(chimera::Command& cmd);

    //CSM debugging
    BOOL SetCascadeViewCamera(chimera::Command& cmd);

    BOOL SetCascadeLightCamera(chimera::Command& cmd);

    BOOL SetCascadeCam0(chimera::Command& cmd);

    BOOL SetCascadeCam1(chimera::Command& cmd);

    BOOL SetCascadeCam2(chimera::Command& cmd);

    BOOL PickActor(chimera::Command& cmd);

    BOOL ApplyPlayerForce(chimera::Command& cmd);

    BOOL RotatXPickedActor(chimera::Command& cmd);
    
    BOOL RotatYPickedActor(chimera::Command& cmd);

    BOOL RotatZPickedActor(chimera::Command& cmd);

    BOOL ToggleRotationDir(chimera::Command& cmd);

    BOOL ApplyForce(chimera::Command& cmd);

    BOOL ApplyTorque(chimera::Command& cmd);

    BOOL SpawnBasicMeshActor(chimera::Command& cmd);

    BOOL SpawnSpotLight(chimera::Command& cmd);

    BOOL ToggleActorPropPhysical(chimera::Command& cmd);

    BOOL DeletePickedActor(chimera::Command& cmd);

    VOID MouseWheelActorPositionModify(INT x, INT y, INT delta);

    BOOL FlushVRam(chimera::Command& cmd);

    VOID MovePicked(VOID);

    VOID ScaleActorAction(FLOAT factor);

    BOOL ScaleActorBigger(chimera::Command& cmd);

    BOOL ScaleActorSmaller(chimera::Command& cmd);

    BOOL Jump(chimera::Command& cmd);

    BOOL ToogleCamera(chimera::Command& cmd);

    BOOL SpawnSpheres(chimera::Command& cmd);

    BOOL SetRasterState(chimera::Command& cmd);

    //set actions here
    VOID RegisterCommands(chimera::ActorController& controller, chimera::CommandInterpreter& interpreter);
}
