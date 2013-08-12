#pragma once
#include "stdafx.h"
#include "GameView.h"
#include "Commands.h"

namespace gameinput
{
    VOID PlayTestSound();

    BOOL SetRenderMode(tbd::Command& cmd);

    BOOL SetDefaultPlayer(tbd::Command& cmd);

    //CSM debugging
    BOOL SetCascadeViewCamera(tbd::Command& cmd);

    BOOL SetCascadeLightCamera(tbd::Command& cmd);

    BOOL SetCascadeCam0(tbd::Command& cmd);

    BOOL SetCascadeCam1(tbd::Command& cmd);

    BOOL SetCascadeCam2(tbd::Command& cmd);

    BOOL PickActor(tbd::Command& cmd);

    BOOL ApplyPlayerForce(tbd::Command& cmd);

    BOOL RotatXPickedActor(tbd::Command& cmd);
    
    BOOL RotatYPickedActor(tbd::Command& cmd);

    BOOL RotatZPickedActor(tbd::Command& cmd);

    BOOL ToggleRotationDir(tbd::Command& cmd);

    BOOL ApplyForce(tbd::Command& cmd);

    BOOL ApplyTorque(tbd::Command& cmd);

    BOOL SpawnBasicMeshActor(tbd::Command& cmd);

    BOOL SpawnSpotLight(tbd::Command& cmd);

    BOOL ToggleActorPropPhysical(tbd::Command& cmd);

    BOOL DeletePickedActor(tbd::Command& cmd);

    VOID MouseWheelActorPositionModify(INT x, INT y, INT delta);

    BOOL FlushVRam(tbd::Command& cmd);

    VOID MovePicked(VOID);

    VOID ScaleActorAction(FLOAT factor);

    BOOL ScaleActorBigger(tbd::Command& cmd);

    BOOL ScaleActorSmaller(tbd::Command& cmd);

    BOOL Jump(tbd::Command& cmd);

    BOOL ToogleCamera(tbd::Command& cmd);

    BOOL SpawnSpheres(tbd::Command& cmd);

    BOOL SetRasterState(tbd::Command& cmd);

    //set actions here
    VOID RegisterCommands(tbd::ActorController& controller, tbd::CommandInterpreter& interpreter);
}
