#pragma once

struct ParticlePosition
{
    float x;
    float y;
    float z;
    float alive;
};

struct EmitterData
{
    float rand;
    float birthTime;
    float time;
    float tmp;
};