#include "ParticleSystem.h"
#include <texture_types.h>
#include "Particles.cuh"
#include "Actor.h"
#include "GameApp.h"
#include "Mesh.h"
#include "Components.h"
#include "EventManager.h"
#include "GameLogic.h"
#include "ActorFactory.h"
#include "FileSystem.h"
#include "util.h"
namespace chimera
{
    namespace gradientpremades
    {

        static int A[] = { 0, 0, 0 };
        static int T[] = { 0x15, 0x38, 0x32, 0x2c, 0x0d, 0x13, 0x07, 0x2a };

        static double u, v, w;
        static int i, j, k;
        static int hi, lo, li, la;
        static double s, t, r, q, p, y, x, z;
        static int b1, b2, b3, b4, b5;
        static int h, pp, hh, gg;

        static int b( int N, int B ) 
        {
            return N >> B & 1;
        }

        static double K(int a) ;
        static int shuffle2( int i, int j, int k );

        static double Noise3( double x, double y, double z ) 
        {
            s = (x + y + z) / 3.;

            i = (int) floor(x + s);
            j = (int) floor(y + s);
            k = (int) floor(z + s);

            s = (i + j + k) / 6.;
            u = x - i + s;
            v = y - j + s;
            w = z - k + s;

            A[0] = A[1] = A[2] = 0;
            hi = u >= w ? u >= v ? 0 : 1 : v >= w ? 1 : 2;
            lo = u < w ? u < v ? 0 : 1 : v < w ? 1 : 2;
            li = (3 - hi - lo);
            la = 0;
            return K(hi) + K(li) + K(lo) + K(la);
        }
        
        static double K(int a) 
        {

            s = (A[0] + A[1] + A[2]) / 6.;

            x = u - A[0] + s;
            y = v - A[1] + s;
            z = w - A[2] + s;
            t = .6 - x * x - y * y - z * z;
            gg = i + A[0];
            hh = j + A[1];
            pp = k + A[2];
            h = shuffle2(gg, hh , pp);
            A[a]++;
            if (t < 0)
                return 0;

            b5 = h >> 5 & 1;
            b4 = h >> 4 & 1;
            b3 = h >> 3 & 1;
            b2 = h >> 2 & 1;
            b1 = h & 3;

            p = b1 == 1 ? x : b1 == 2 ? y : z;
            q = b1 == 1 ? y : b1 == 2 ? z : x;
            r = b1 == 1 ? z : b1 == 2 ? x : y;

            p = (b5 == b3 ? -p : p);
            q = (b5 == b4 ? -q : q);
            r = (b5 != (b4 ^ b3) ? -r : r);
            t *= t;
            return 8 * t * t * (p + (b1 == 0 ? q + r : b2 == 0 ? q : r));
        }

        static int shuffle2( int i, int j, int k ) 
        {
            return 
                T[4 * (i >> 0 & 1) + 2 * (j >> 0 & 1) + (k >> 0 & 1)] + 
                T[4 * (j >> 1 & 1) + 2 * (k >> 1 & 1) + (i >> 1 & 1)] + 
                T[4 * (k >> 2 & 1) + 2 * (i >> 2 & 1) + (j >> 2 & 1)] + 
                T[4 * (i >> 3 & 1) + 2 * (j >> 3 & 1) + (k >> 3 & 1)] + 
                T[4 * (j >> 4 & 1) + 2 * (k >> 4 & 1) + (i >> 4 & 1)] + 
                T[4 * (k >> 5 & 1) + 2 * (i >> 5 & 1) + (j >> 5 & 1)] + 
                T[4 * (i >> 6 & 1) + 2 * (j >> 6 & 1) + (k >> 6 & 1)] +
                T[4 * (j >> 7 & 1) + 2 * (k >> 7 & 1) + (i >> 7 & 1)];
        }

        static float* GenerateNoiseValues(int width, int height, int depth, float amplitude, float frequency) 
        {
            float* buffer = new float[width * height * depth * 4];
            float n[] = { 0, 0, 0, 1 };
            int format = 4;
            int count = 0;
            float incz, incx, incy, _amp, _frequency;

            float r = 0;
            for (int z = 0; z < depth; z++ ) 
            {

                for (int x = 0; x < height; x++ ) 
                {

                    for (int y = 0; y < width; y++) 
                    {
                        _amp = amplitude;
                        _frequency = 16;
                        FLOAT v = 0;
                        for (int o = 0; o < format; o++, _amp *= 0.5, _frequency *= 2) 
                        {
                            incx = 1.0f / ((float)width / _frequency);
                            incz = 1.0f / ((float)depth / _frequency);
                            incy = 1.0f / ((float)height / _frequency);

                            n[0] = incz * z;
                            n[1] = incx * x;
                            n[2] = incy * y;

                            r = (float)Noise3(n[0], n[1], n[2]);
                            v += r;
                        }
                        _amp = amplitude;
                        _frequency = 17;
                        buffer[count++] = v;
                        v = 0;
                        for (int o = 0; o < format; o++, _amp *= 0.5, _frequency *= 2) 
                        {
                            incx = 1.0f / ((float)width / _frequency);
                            incz = 1.0f / ((float)depth / _frequency);
                            incy = 1.0f / ((float)height / _frequency);

                            n[0] = incz * z;
                            n[1] = incx * x;
                            n[2] = incy * y;

                            r = (float)Noise3(n[0], n[1], n[2]);
                            v += r;
                        }
                        _amp = amplitude;
                        _frequency = 18;
                        buffer[count++] = v;
                        v = 0;
                        for (int o = 0; o < format; o++, _amp *= 0.5, _frequency *= 2) 
                        {
                            incx = 1.0f / ((float)width / _frequency);
                            incz = 1.0f / ((float)depth / _frequency);
                            incy = 1.0f / ((float)height / _frequency);

                            n[0] = incz * z;
                            n[1] = incx * x;
                            n[2] = incy * y;

                            r = (float)Noise3(n[0], n[1], n[2]);
                            v += r;
                        }
                        buffer[count++] = v;
                        buffer[count++] = 0;
                    }
                }
            }
            return buffer;
        }

        FLOAT* RandomStuff0(UINT w, UINT h, UINT d, FLOAT scale)
        {
            UINT size = 4;
            FLOAT* data = new FLOAT[w * h * d * size];
            for(UINT y = 0; y < d; ++y)
            {
                for(UINT z = 0; z < h; ++z)
                {
                    for(UINT x = 0; x < w; ++x)
                    {
                        UINT pos = size * (z * w * h + y * w + x);
                        //if(x * x + z * z <  radius * radius)
                        {
                            FLOAT sx = cos(XM_PI * z / (FLOAT)h) * sin(XM_PI *  (w-x) / (FLOAT)w);
                            FLOAT sz = (1 - sin(XM_PI * z / (FLOAT)h)) * cos(XM_PI *  (w-x) / (FLOAT)w);
                            FLOAT sy = cos(XM_PI * y / (FLOAT)h) * sin(XM_PI *  (w-x) / (FLOAT)w);
                            util::Vec3 dir(sx, sy, sz);
                            //dir.Normalize();
                            data[pos + 0] = scale * dir.x;
                            data[pos + 1] = scale * dir.y;
                            data[pos + 2] = scale * dir.z;
                            data[pos + 3] = 0;
                        }
                    }
                }
            }
            return data;
        }

        FLOAT* RandomStuff1(UINT w, UINT h, UINT d, FLOAT scale)
        {
            UINT size = 4;
            FLOAT* data = new FLOAT[w * h * d * size];
            for(UINT y = 0; y < d; ++y)
            {
                for(UINT z = 0; z < h; ++z)
                {
                    for(UINT x = 0; x < w; ++x)
                    {
                        UINT pos = size * (z * w * h + y * w + x);
                        //if(x * x + z * z <  radius * radius)
                        {
                            FLOAT sx = cos(x / (FLOAT)w * XM_2PI) * sin(z / (FLOAT)w * XM_2PI);
                            FLOAT sy = cos((x+z) / (FLOAT)h * XM_2PI) * sin((x+z) / (FLOAT)w * XM_2PI);
                            FLOAT sz = cos(z / (FLOAT)d * XM_2PI) * sin(x / (FLOAT)w * XM_2PI);
                            util::Vec3 dir(sx, sy, sz);
                            //dir.Normalize();
                            data[pos + 0] = scale * dir.x;
                            data[pos + 1] = scale * dir.y;
                            data[pos + 2] = scale * dir.z;
                            data[pos + 3] = 0;
                        }
                    }
                }
            }
            return data;
        }

        FLOAT* RandomStuff2(UINT w, UINT h, UINT d, FLOAT scale)
        {
            UINT size = 4;
            FLOAT* data = new FLOAT[w * h * d * size];
            for(UINT y = 0; y < d; ++y)
            {
                for(UINT z = 0; z < h; ++z)
                {
                    for(UINT x = 0; x < w; ++x)
                    {
                        UINT pos = size * (z * w * h + y * w + x);
                        //if(x * x + z * z <  radius * radius)
                        {
                            FLOAT sx = cos((XM_2PI * x / (FLOAT)w)) + sin((XM_2PI * x / (FLOAT)w));
                            FLOAT sy = sin((XM_2PI * x / (FLOAT)h)) + cos((XM_2PI * x / (FLOAT)w));
                            FLOAT sz = cos((XM_2PI * z / (FLOAT)d)) + sin((XM_2PI * z / (FLOAT)w));
                            util::Vec3 dir(sx, sy, sz);
                            data[pos + 0] = scale * dir.x;
                            data[pos + 1] = scale * dir.y;
                            data[pos + 2] = scale * dir.z;
                            data[pos + 3] = 0;
                        }
                    }
                }
            }
            return data;
        }

        FLOAT fx(FLOAT x, FLOAT y, FLOAT z, FLOAT amp)
        {
            FLOAT a = sin(x);
            FLOAT b = cos(y);
            FLOAT c = sin(z);
            return amp * (a * a + b * b + c * c);
        }

        FLOAT* RandomStuff3(UINT w, UINT h, UINT d, FLOAT scale)
        {
            UINT size = 4;
            FLOAT freq = 1;
            FLOAT amp = 0.1f;
            FLOAT* data = new FLOAT[w * h * d * size];
            FLOAT time = 10;
            FLOAT delta = 1.0f / (FLOAT)w;
            for(UINT y = 0; y < d; ++y)
            {
                for(UINT z = 0; z < h; ++z)
                {
                    for(UINT x = 0; x < w; ++x)
                    {
                        UINT pos = size * (z * w * h + y * w + x);
                        //if(x * x + z * z <  radius * radius)
                        {
                            FLOAT r = y / (FLOAT)(d-1);
                            FLOAT phi = -XM_PI + 2 * x / (FLOAT)(w-1) * XM_PI;
                            FLOAT theta = z / (FLOAT)(h-1) * XM_PI;
   
                            FLOAT sx = r * sin(theta) * cos(phi);
                            FLOAT sy = r * cos(theta);
                            FLOAT sz = r * sin(theta) * sin(phi);

                            FLOAT sx1 = r * sin(theta + r * delta) * cos(phi + r * delta);
                            FLOAT sy1 = r * cos(theta + r * delta);
                            FLOAT sz1 = r * sin(theta + r * delta) * sin(phi + r * delta);

                            util::Vec3 dir(sx1 - sx, sy1 - sy, sz1 - sz);
                            dir.Normalize();
                            data[pos + 0] = dir.x;
                            data[pos + 1] = dir.y;
                            data[pos + 2] = dir.z;
                            data[pos + 3] = 0;
                        }
                    }
                }
            }
            return data;
        }
    }

    ActorBasedModifier::ActorBasedModifier(VOID)
    {
        chimera::EventListener listener = fastdelegate::MakeDelegate(this, &ActorBasedModifier::ActorMovedDelegate);
        chimera::IEventManager::Get()->VAddEventListener(listener, chimera::ActorMovedEvent::TYPE);
    }

    std::shared_ptr<Actor> ActorBasedModifier::CreateModActor(CONST util::Vec3& pos, LPCSTR info, CONST FLOAT scale)
    {
        chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();
        chimera::RenderComponent* cmp = desc->AddComponent<chimera::RenderComponent>(chimera::RenderComponent::COMPONENT_ID);
        cmp->m_type = "anchor";
        cmp->m_info = info;

        chimera::TransformComponent* tc = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);

        tc->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

        tc->GetTransformation()->SetScale(scale);

        desc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);

        std::shared_ptr<chimera::Actor> actor = chimera::g_pApp->GetLogic()->VCreateActor(desc);

        m_actorId = actor->GetId();

        return actor;
    }

    VOID ActorBasedModifier::ActorMovedDelegate(event::IEventPtr pEventData)
    {
        std::shared_ptr<event::ActorMovedEvent> event = std::static_pointer_cast<event::ActorMovedEvent>(pEventData);
        if(event->m_actor->GetId() == m_actorId)
        {
            VOnActorMoved(event->m_actor);
        }
    }

    ActorBasedModifier::~ActorBasedModifier(VOID)
    {
        event::IEventPtr event(new event::DeleteActorEvent(m_actorId));
        event::IEventManager::Get()->VQueueEvent(event);

        event::EventListener listener = fastdelegate::MakeDelegate(this, &ActorBasedModifier::ActorMovedDelegate);
        event::IEventManager::Get()->VRemoveEventListener(listener, event::ActorMovedEvent::TYPE);
    }

    BaseEmitter::BaseEmitter(CONST util::Vec3& position, UINT count, FLOAT starSpawnTime, FLOAT endSpawnTime) 
        : m_particleCount(count), m_startSpawnTime(starSpawnTime), m_endSpawnTime(endSpawnTime), m_position(position)
    {

    }

    VOID BaseEmitter::VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt)
    {
        float3 translation;
        translation.x = sys->GetTranslation().x;
        translation.y = sys->GetTranslation().y;
        translation.z = sys->GetTranslation().z;

        /*
        computeEmitter(
            (float4*)sys->GetParticles()->ptr,
            (float4*)m_startingPositions->ptr,
            (float3*)sys->GetVelocities()->ptr, 
            (float3*)sys->GetAcceleration()->ptr,
            (EmitterData*)m_emitterData->ptr,
            translation, (FLOAT)time, (FLOAT)dt, (FLOAT)m_startSpawnTime, (FLOAT)m_endSpawnTime, m_particleCount, sys->GetLocalWorkSize()); */

        INT threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
        m_kernel->m_blockDim.x = sys->GetLocalWorkSize();
        m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;

        VOID *args[] = 
        { 
            &sys->GetParticles()->ptr,
            &m_startingPositions->ptr, &sys->GetVelocities()->ptr, &sys->GetAcceleration()->ptr, &m_emitterData->ptr, &translation,
            &time, &dt, &m_startSpawnTime, &m_endSpawnTime, &m_particleCount
        };

        m_kernel->m_ppArgs = args;
        sys->GetCuda()->CallKernel(m_kernel);

    }

    VOID BaseEmitter::VOnRestore(ParticleSystem* sys)
    {
        cudah::cudah* cuda = sys->GetCuda();
        m_kernel = cuda->GetKernel("_computeEmitter");

        PositiveNormalizedUniformValueGenerator rg(m_particleCount, 10);

        FLOAT* rands = rg.CreateRandomValues();
        EmitterData* data = new EmitterData[m_particleCount];
        for(UINT i = 0; i < m_particleCount; ++i)
        {
            data[i].rand = rands[i];
            data[i].time = 0;
            data[i].birthTime = 0;
            data[i].tmp = 0;
        }
        m_emitterData = cuda->CreateBuffer(std::string("EmitterData"), m_particleCount * 16, data, 16);
        SAFE_ARRAY_DELETE(rands);
        SAFE_ARRAY_DELETE(data);

        ParticlePosition* positions = CreateParticles();
        m_startingPositions = cuda->CreateBuffer(std::string("StartingPositions"), m_particleCount * 16, positions, 16);

        SAFE_ARRAY_DELETE(positions);
    }


    VOID BaseModifier::SetAABB(CONST util::AxisAlignedBB& aabb)
    {
        m_aabb = aabb;
    }

    PointEmitter::PointEmitter(CONST util::Vec3& position, UINT particleCount, FLOAT start, FLOAT end) : BaseEmitter(position, particleCount, start, end)
    {

    }

    ParticlePosition* PointEmitter::CreateParticles(VOID)
    {
        ParticlePosition* positions = new ParticlePosition[m_particleCount];
        for(UINT i = 0; i < m_particleCount; ++i)
        {
            positions[i].x = m_position.x;
            positions[i].y = m_position.y;
            positions[i].z = m_position.z;
            positions[i].w = 0;
        }
        return positions;
    }

    BoxEmitter::BoxEmitter(CONST util::Vec3& extends, CONST util::Vec3& position, UINT particleCount, FLOAT start, FLOAT end) 
        : BaseEmitter(position, particleCount, start, end), m_extends(extends)
    {

    }

    ParticlePosition* BoxEmitter::CreateParticles(VOID)
    {
        ParticlePosition* positions = new ParticlePosition[m_particleCount];
        srand(2);
        for(UINT i = 0; i < m_particleCount; ++i)
        {
            positions[i].x = m_position.x - m_extends.x + 2 * m_extends.x * rand() / (FLOAT)RAND_MAX;
            positions[i].y = m_position.y - m_extends.y + 2 * m_extends.y * rand() / (FLOAT)RAND_MAX;;
            positions[i].z = m_position.z - m_extends.z + 2 * m_extends.z * rand() / (FLOAT)RAND_MAX;;
            positions[i].w = 0;
        }
        return positions;
    }

    SurfaceEmitter::SurfaceEmitter(chimera::CMResource meshFile, CONST util::Vec3& position, UINT particleCount, FLOAT start, FLOAT end) 
        : BaseEmitter(position, particleCount, start, end), m_meshFile(meshFile)
    {

    }

    util::Vec3 GetVertex(std::shared_ptr<chimera::Mesh> mesh, UINT index)
    {
        UINT stride = mesh->GetVertexStride() / 4;
        FLOAT x = mesh->GetVertices()[index * stride + 0];
        FLOAT y = mesh->GetVertices()[index * stride + 1];
        FLOAT z = mesh->GetVertices()[index * stride + 2];
        return util::Vec3(x, y, z);
    }

    ParticlePosition* SurfaceEmitter::CreateParticles(VOID)
    {
        std::shared_ptr<chimera::Mesh> mesh = std::static_pointer_cast<chimera::Mesh>(chimera::g_pApp->GetCache()->GetHandle(m_meshFile));
        CONST std::list<chimera::Face>& faces = mesh->GetFaces();
        ParticlePosition* poses = new ParticlePosition[m_particleCount];
        srand(10);
        for(UINT i = 0; i < m_particleCount; ++i)
        {
            util::Vec3 v0;
            util::Vec3 v1;
            util::Vec3 v2;

            INT index = (INT)((rand() / (FLOAT)RAND_MAX) * (mesh->GetFaces().size() - 1));
            //DEBUG_OUT_A("%d, %d", index, mesh->GetFaces().size());
            std::list<Face>::const_iterator it = faces.begin();
            std::advance(it, index);
            Face f = *it;

            if(f.m_triples.size() == 3)
            {
                UINT iv0 = f.m_triples[0].position;
                UINT iv1 = f.m_triples[1].position;
                UINT iv2 = f.m_triples[2].position;

                v0 = GetVertex(mesh, iv0);
                v1 = GetVertex(mesh, iv1);
                v2 = GetVertex(mesh, iv2);
            }
            else if(f.m_triples.size() == 4)
            {
                UINT iv0 = f.m_triples[0].position;
                UINT iv1 = f.m_triples[1].position;
                UINT iv2 = f.m_triples[2].position;
                UINT iv3 = f.m_triples[3].position;
                if(rand() / (FLOAT) RAND_MAX < 0.5)
                {
                    v0 = GetVertex(mesh, iv0);
                    v1 = GetVertex(mesh, iv1);
                    v2 = GetVertex(mesh, iv2);
                }
                else
                {
                    v0 = GetVertex(mesh, iv1);
                    v1 = GetVertex(mesh, iv2);
                    v2 = GetVertex(mesh, iv3);
                }
            }
            else
            {
                LOG_CRITICAL_ERROR("SurfaceEmitter error, unknown triples size");
            }

            FLOAT a = rand() / (FLOAT)RAND_MAX;
            FLOAT b = (1-a) * rand() / (FLOAT)RAND_MAX;
            FLOAT c = 1 - a - b;

            util::Vec3 pos = v0 * a + v1 * b + v2 * c;
            ParticlePosition p(m_position.x + pos.x, m_position.y + pos.y, m_position.z + pos.z, 0);
            poses[i] = p;
        }

        return poses;
    }

    Gravity::Gravity(FLOAT factor, UCHAR axis) : m_axis(axis) , m_factor(factor)
    {

    }

    VOID Gravity::VOnRestore(ParticleSystem* sys)
    {
        m_kernel = sys->GetCuda()->GetKernel("_computeGravity");
    }

    VOID Gravity::VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt)
    {
        //computeGravity((float4*)sys->GetParticles()->ptr, (float3*)sys->GetAcceleration()->ptr, m_factor, sys->GetParticlesCount(), sys->GetLocalWorkSize()); 

        INT threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
        m_kernel->m_blockDim.x = sys->GetLocalWorkSize();
        m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;

        VOID *args[] = 
        { 
            &sys->GetParticles()->ptr, &sys->GetAcceleration()->ptr, &m_factor
        };

        m_kernel->m_ppArgs = args;

        sys->GetCuda()->CallKernel(m_kernel);
    }

    VOID Turbulence::VOnRestore(ParticleSystem* sys)
    {
        UINT count = 256 * 3;
        NormalizedUniformValueGenerator rg(count, m_seed, m_strength);
        FLOAT* vs = rg.CreateRandomValues();
        m_randomDirections = sys->GetCuda()->CreateBuffer(std::string("turbulenceDirs"), 256 * 3 * sizeof(FLOAT), vs, sizeof(FLOAT));
        m_kernel = sys->GetCuda()->GetKernel("_computeTurbulence");
        SAFE_ARRAY_DELETE(vs);
    }

    VOID Turbulence::VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt)
    {
        //computeTurbulence((float4*)sys->GetParticles()->ptr, (float3*)sys->GetAcceleration()->ptr, (float3*)m_randomDirections->ptr, 256, time, sys->GetParticlesCount(), sys->GetLocalWorkSize());
        INT threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
        m_kernel->m_blockDim.x = sys->GetLocalWorkSize();
        m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;
        
        INT rngCnt = 256;

        VOID *args[] = 
        { 
            &sys->GetParticles()->ptr, &sys->GetAcceleration()->ptr, &m_randomDirections->ptr, &rngCnt, &time
        };

        m_kernel->m_ppArgs = args;

        sys->GetCuda()->CallKernel(m_kernel);
    }

    UINT Turbulence::VGetByteCount(VOID)
    {
        return 256 * 3 * sizeof(FLOAT);
    }

    typedef FLOAT* (*PROC)(UINT,UINT,UINT,FLOAT);

    GradientField::GradientField(VOID)
    {
        FLOAT scale = 0.1f;

        m_array = NULL;

        m_positionNscale.x = m_positionNscale.y = m_positionNscale.z = 0, m_positionNscale.w = scale;

        CreateModActor(util::Vec3(0,0,0), "GradientField", scale);
    }

    VOID GradientField::VOnRestore(ParticleSystem* sys)
    {
        UINT w = 64;
        UINT h = 64;
        UINT d = 64;
        UINT size = 4;

        chimera::DLL dll("../../ParticleData/ParticleData/x64/Debug/ParticleData.dll");
        FLOAT* data = dll.GetFunction<PROC>("CreateData")(w,h,d,1);
        //FLOAT* data = gradientpremades::RandomStuff2(w, h, d, 0.1f); //2, 16);//
        //FLOAT* data = gradientpremades::GenerateNoiseValues(w, h, d, 2, 16);

        std::stringstream ss;
        ss << "gradientTexture_";
        ss << m_actorId;
        if(m_array)
        {
            sys->GetCuda()->DeleteArray(m_array);
        }

        m_array = sys->GetCuda()->CreateArray(ss.str(), w, h, d, cudah::eRGBA, data);


        //bindGradientTexture(array->m_array, array->GetChannelDesc());
        /*
        ct_gradientTexture.addressMode[0] = cudaAddressModeMirror;
        ct_gradientTexture.addressMode[1] = cudaAddressModeMirror;
        ct_gradientTexture.addressMode[2] = cudaAddressModeMirror;
        ct_gradientTexture.normalized = 1;
        ct_gradientTexture.filterMode = cudaFilterModeLinear;*/
        /*
        CUtexref ref = sys->GetCuda()->GetTexRef("ct_gradientTexture");
        
        CUDA_DRIVER_SAFE_CALLING(cuTexRefSetArray(ref, array->m_array, CU_TRSA_OVERRIDE_FORMAT));
        CUDA_DRIVER_SAFE_CALLING(cuTexRefSetFlags(ref, CU_TRSF_NORMALIZED_COORDINATES));
        CUDA_DRIVER_SAFE_CALLING(cuTexRefSetFilterMode(ref, CU_TR_FILTER_MODE_LINEAR));
        CUDA_DRIVER_SAFE_CALLING(cuTexRefSetAddressMode(ref, 0, CU_TR_ADDRESS_MODE_MIRROR));
        */
        sys->GetCuda()->BindArrayToTexture("ct_gradientTexture", m_array, CU_TR_FILTER_MODE_LINEAR, CU_TR_ADDRESS_MODE_MIRROR, 0);

        m_kernel = sys->GetCuda()->GetKernel("_computeGradientField");

        SAFE_ARRAY_DELETE(data);
    }

    UINT GradientField::VGetByteCount(VOID)
    {
        return 64 * 64 * 64 * 16;
    }

    VOID GradientField::VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt)
    {
        //computeGradientField((float4*)sys->GetParticles()->ptr, (float3*)sys->GetVelocities()->ptr, m_positionNscale, sys->GetParticlesCount(), sys->GetLocalWorkSize());
        INT threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
        m_kernel->m_blockDim.x = sys->GetLocalWorkSize();
        m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;

        VOID *args[] = 
        { 
            &sys->GetParticles()->ptr, &sys->GetVelocities()->ptr, &m_positionNscale
        };

        m_kernel->m_ppArgs = args;

        sys->GetCuda()->CallKernel(m_kernel);
    }

    VOID GradientField::VOnActorMoved(std::shared_ptr<chimera::Actor> actor)
    {
        std::shared_ptr<chimera::TransformComponent> cmp = actor->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock();
        m_positionNscale.x = cmp->GetTransformation()->GetTranslation().x;
        m_positionNscale.y = cmp->GetTransformation()->GetTranslation().y;
        m_positionNscale.z = cmp->GetTransformation()->GetTranslation().z;
        m_positionNscale.w = cmp->GetTransformation()->GetScale().x;
    }

    GravityField::GravityField(CONST util::Vec3& position, CONST FLOAT range, CONST FLOAT scale, GravityPolarization pole)
    {
        m_posistionNrange.x = position.x;
        m_posistionNrange.y = position.y;
        m_posistionNrange.z = position.z;
        m_posistionNrange.w = range;
        m_pole = pole;
        m_scale = scale;
        CreateModActor(position, "GravityField", range);
    }

    VOID GravityField::VOnRestore(ParticleSystem* sys)
    {
        m_kernel = sys->GetCuda()->GetKernel("_computeGravityField");
    }

    VOID GravityField::VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt)
    {
        //computeGravityField((float4*)sys->GetParticles()->ptr, (float3*)sys->GetVelocities()->ptr, m_posistionNrange, m_pole == eRepel, m_scale, sys->GetParticlesCount(), sys->GetLocalWorkSize());
        INT threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
        m_kernel->m_blockDim.x = sys->GetLocalWorkSize();
        m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;

        INT i = m_pole == eRepel;

        VOID *args[] = 
        { 
            &sys->GetParticles()->ptr, &sys->GetVelocities()->ptr, &m_posistionNrange, &i, &m_scale
        };

        m_kernel->m_ppArgs = args;

        sys->GetCuda()->CallKernel(m_kernel);
    }

    VOID GravityField::VOnActorMoved(std::shared_ptr<chimera::Actor> actor)
    {
        std::shared_ptr<chimera::TransformComponent> cmp = actor->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock();
        m_posistionNrange.x = cmp->GetTransformation()->GetTranslation().x;
        m_posistionNrange.y = cmp->GetTransformation()->GetTranslation().y;
        m_posistionNrange.z = cmp->GetTransformation()->GetTranslation().z;
        m_posistionNrange.w = cmp->GetTransformation()->GetScale().x;
    }

    VOID VelocityDamper::VOnRestore(ParticleSystem* sys)
    {
        m_kernel = sys->GetCuda()->GetKernel("_computeVelocityDamping");
    }

    VOID VelocityDamper::VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt)
    {
        //computeVelocityDamping((float4*)sys->GetParticles()->ptr, (float3*)sys->GetVelocities()->ptr, m_dampValue,  sys->GetParticlesCount(), sys->GetLocalWorkSize());
        INT threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
        m_kernel->m_blockDim.x = sys->GetLocalWorkSize();
        m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;

        VOID *args[] = 
        { 
            &sys->GetParticles()->ptr, &sys->GetVelocities()->ptr, &m_dampValue
        };

        m_kernel->m_ppArgs = args;

        sys->GetCuda()->CallKernel(m_kernel);
    }

    VOID Plane::VOnRestore(ParticleSystem* sys)
    {
        m_kernel = sys->GetCuda()->GetKernel("_computePlane");
    }

    VOID Plane::VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt)
    {
        INT threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
        m_kernel->m_blockDim.x = sys->GetLocalWorkSize();
        m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;

        DirectX::XMFLOAT4 n;
        n.x = m_plane.GetNormal().m_v.x;
        n.y = m_plane.GetNormal().m_v.y;
        n.z = m_plane.GetNormal().m_v.z;
        n.w = m_plane.GetRadius();

        VOID *args[] = 
        { 
            &n, &sys->GetVelocities()->ptr, &sys->GetParticles()->ptr,
        };

        m_kernel->m_ppArgs = args;

        sys->GetCuda()->CallKernel(m_kernel);
    }

    BoundingBox::BoundingBox(VOID)
    {
        m_pData[0] = NULL;
        m_pData[1] = NULL;
    }

    BoundingBox::~BoundingBox(VOID)
    {
        SAFE_ARRAY_DELETE(m_pData[0]);
        SAFE_ARRAY_DELETE(m_pData[1]);
    }

    VOID BoundingBox::VOnRestore(ParticleSystem* sys)
    {
        SAFE_ARRAY_DELETE(m_pData[0]);
        SAFE_ARRAY_DELETE(m_pData[1]);
        m_pData[0] = new FLOAT[3 * 1024];
        m_pData[1] = new FLOAT[3 * 1024];
        m_kernel = sys->GetCuda()->GetKernel("_reduce_max4");
        m_second = sys->GetCuda()->GetKernel("_reduce_min4");
        m_min = sys->GetCuda()->CreateBuffer(std::string("__min"), 1024 * 3 * sizeof(FLOAT), 3 * sizeof(FLOAT));
        m_max = sys->GetCuda()->CreateBuffer(std::string("__max"), 1024 * 3 * sizeof(FLOAT), 3 * sizeof(FLOAT));
    }

    VOID _GetAABB(float3* data[2], UINT l, util::AxisAlignedBB& aabb)
    {
        aabb.Clear();
        for(UINT i = 0; i < l; ++i)
        {
            float3 m0 = data[0][i];
            float3 m1 = data[1][i];
            aabb.AddPoint(util::Vec3(m0.x, m0.y, m0.z));
            aabb.AddPoint(util::Vec3(m1.x, m1.y, m1.z));
        }
        aabb.Construct();
    }

    VOID BoundingBox::VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt)
    {
        INT threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
        m_kernel->m_blockDim.x = 512;//sys->GetLocalWorkSize();
        m_kernel->m_gridDim.x = (threads/2) / m_kernel->m_blockDim.x;
        m_kernel->m_shrdMemBytes = 3 * (m_kernel->m_blockDim.x) * sizeof(FLOAT);
        VOID *args[] = 
        { 
            &sys->GetParticles()->ptr, &m_max->ptr
        };

        m_kernel->m_ppArgs = args;

        sys->GetCuda()->CallKernel(m_kernel);

        m_second->m_blockDim.x = m_kernel->m_blockDim.x;
        m_second->m_gridDim.x = m_kernel->m_gridDim.x;
        m_second->m_shrdMemBytes = m_kernel->m_shrdMemBytes;

        VOID* args2[] = 
        { 
            &sys->GetParticles()->ptr, &m_min->ptr
        };

        m_second->m_ppArgs = args2;

        sys->GetCuda()->CallKernel(m_second);
        
        sys->GetCuda()->ReadBuffer(m_max, m_pData[0]);

        sys->GetCuda()->ReadBuffer(m_min, m_pData[1]);

        _GetAABB((float3**)m_pData, 1024, m_aabb);

        sys->SetAxisAlignedBB(m_aabb);
    }

    UINT BoundingBox::VGetByteCount(VOID)
    {
        return m_min->GetByteSize() + m_max->GetByteSize();
    }

    KDTree::KDTree(VOID) : m_pBB(NULL)
    {

    }

    VOID KDTree::VOnRestore(ParticleSystem* sys)
    {
        if(!m_pBB)
        {
            m_pBB = new BoundingBox();
        }
        m_pBB->VOnRestore(sys);

    }

    VOID KDTree::VUpdate(ParticleSystem* sys, FLOAT time, FLOAT dt)
    {
        m_pBB->VUpdate(sys, time, dt);
    }

    UINT KDTree::VGetByteCount(VOID)
    {
        return 0;
    }

    KDTree::~KDTree(VOID)
    {
        SAFE_DELETE(m_pBB);
    }
}