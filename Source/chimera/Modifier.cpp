#include "ParticleSystem.h"
#include <cuda/cuda_helper.h>
#include "Event.h"

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

        static float* GenerateNoiseValues(int width, int height, int depth, float amplitude, float frequency, float* buffer) 
        {
            //float* buffer = new float[width * height * depth * 4];
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
                        float v = 0;
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

        float* RandomStuff0(uint w, uint h, uint d, float scale, float* data)
        {
            uint size = 4;
            //float* data = new float[w * h * d * size];
            for(uint y = 0; y < d; ++y)
            {
                for(uint z = 0; z < h; ++z)
                {
                    for(uint x = 0; x < w; ++x)
                    {
                        uint pos = size * (z * w * h + y * w + x);
                        //if(x * x + z * z <  radius * radius)
                        {
                            float sx = cos(XM_PI * z / (float)h) * sin(XM_PI *  (w-x) / (float)w);
                            float sz = (1 - sin(XM_PI * z / (float)h)) * cos(XM_PI *  (w-x) / (float)w);
                            float sy = cos(XM_PI * y / (float)h) * sin(XM_PI *  (w-x) / (float)w);
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

        float* RandomStuff1(uint w, uint h, uint d, float scale)
        {
            uint size = 4;
            float* data = new float[w * h * d * size];
            for(uint y = 0; y < d; ++y)
            {
                for(uint z = 0; z < h; ++z)
                {
                    for(uint x = 0; x < w; ++x)
                    {
                        uint pos = size * (z * w * h + y * w + x);
                        //if(x * x + z * z <  radius * radius)
                        {
                            float sx = cos(x / (float)w * XM_2PI) * sin(z / (float)w * XM_2PI);
                            float sy = cos((x+z) / (float)h * XM_2PI) * sin((x+z) / (float)w * XM_2PI);
                            float sz = cos(z / (float)d * XM_2PI) * sin(x / (float)w * XM_2PI);
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

        float* RandomStuff2(uint w, uint h, uint d, float scale)
        {
            uint size = 4;
            float* data = new float[w * h * d * size];
            for(uint y = 0; y < d; ++y)
            {
                for(uint z = 0; z < h; ++z)
                {
                    for(uint x = 0; x < w; ++x)
                    {
                        uint pos = size * (z * w * h + y * w + x);
                        //if(x * x + z * z <  radius * radius)
                        {
                            float sx = cos((XM_2PI * x / (float)w)) + sin((XM_2PI * x / (float)w));
                            float sy = sin((XM_2PI * x / (float)h)) + cos((XM_2PI * x / (float)w));
                            float sz = cos((XM_2PI * z / (float)d)) + sin((XM_2PI * z / (float)w));
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

        float fx(float x, float y, float z, float amp)
        {
            float a = sin(x);
            float b = cos(y);
            float c = sin(z);
            return amp * (a * a + b * b + c * c);
        }

        float* RandomStuff3(uint w, uint h, uint d, float scale, float* data)
        {
            uint size = 4;
            float freq = 1;
            float amp = 0.1f;
            float time = 10;
            float delta = 1.0f / (float)w;
            for(uint y = 0; y < d; ++y)
            {
                for(uint z = 0; z < h; ++z)
                {
                    for(uint x = 0; x < w; ++x)
                    {
                        uint pos = size * (z * w * h + y * w + x);
                        //if(x * x + z * z <  radius * radius)
                        {
                            float r = y / (float)(d-1);
                            float phi = -XM_PI + 2 * x / (float)(w-1) * XM_PI;
                            float theta = z / (float)(h-1) * XM_PI;
   
                            float sx = r * sin(theta) * cos(phi);
                            float sy = r * cos(theta);
                            float sz = r * sin(theta) * sin(phi);

                            float sx1 = r * sin(theta + r * delta) * cos(phi + r * delta);
                            float sy1 = r * cos(theta + r * delta);
                            float sz1 = r * sin(theta + r * delta) * sin(phi + r * delta);

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

//     ActorBasedModifier::ActorBasedModifier(void)
//     {
//         chimera::EventListener listener = fastdelegate::MakeDelegate(this, &ActorBasedModifier::ActorMovedDelegate);
//         chimera::IEventManager::Get()->VAddEventListener(listener, chimera::ActorMovedEvent::TYPE);
//     }
// 
//     std::shared_ptr<Actor> ActorBasedModifier::CreateModActor(const util::Vec3& pos, LPCSTR info, const float scale)
//     {
//         chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();
//         chimera::RenderComponent* cmp = desc->AddComponent<chimera::RenderComponent>(chimera::RenderComponent::COMPONENT_ID);
//         cmp->m_type = "anchor";
//         cmp->m_info = info;
// 
//         chimera::TransformComponent* tc = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);
// 
//         tc->GetTransformation()->SetTranslation(pos.x, pos.y, pos.z);
// 
//         tc->GetTransformation()->SetScale(scale);
// 
//         desc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);
// 
//         std::shared_ptr<chimera::Actor> actor = chimera::g_pApp->GetLogic()->VCreateActor(desc);
// 
//         m_actorId = actor->GetId();
// 
//         return actor;
//     }
// 
//     void ActorBasedModifier::ActorMovedDelegate(event::IEventPtr pEventData)
//     {
//         std::shared_ptr<event::ActorMovedEvent> event = std::static_pointer_cast<event::ActorMovedEvent>(pEventData);
//         if(event->m_actor->GetId() == m_actorId)
//         {
//             VOnActorMoved(event->m_actor);
//         }
//     }
// 
//     ActorBasedModifier::~ActorBasedModifier(void)
//     {
//         event::IEventPtr event(new event::DeleteActorEvent(m_actorId));
//         event::IEventManager::Get()->VQueueEvent(event);
// 
//         event::EventListener listener = fastdelegate::MakeDelegate(this, &ActorBasedModifier::ActorMovedDelegate);
//         event::IEventManager::Get()->VRemoveEventListener(listener, event::ActorMovedEvent::TYPE);
//     }
// 
// 
    Gravity::Gravity(const util::Vec3& dir)
    {
        m_dir = dir;
    }

    void Gravity::VOnRestore(IParticleSystem* sys, IParticleEmitter* emitter)
    {
        m_module.Create(KERNEL_PTX);
        m_kernel.Create(m_module.GetFunction("_computeGravity"));
    }

    void Gravity::VUpdate(IParticleSystem* sys, IParticleEmitter* emitter, float time, float dt)
    {
        uint N = emitter->VGetParticleCount();
        uint grid = nutty::cuda::GetCudaGrid(N, CUDA_GROUP_SIZE);
        m_kernel.SetDimension(grid, CUDA_GROUP_SIZE);

        void* ptr = emitter->VGetParticleArray();
        void* accel = emitter->VGetAccelerationArray();
        m_kernel.SetRawKernelArg(0, &ptr);
        m_kernel.SetRawKernelArg(1, &accel);
        m_kernel.SetKernelArg(2, m_dir.m_v);
        m_kernel.SetKernelArg(3, N);

        m_kernel.Call();
    }
// 
    void Turbulence::VOnRestore(IParticleSystem* sys, IParticleEmitter* emitter)
    {
        m_module.Create(KERNEL_PTX);
        m_kernel.Create(m_module.GetFunction("_computeTurbulence"));

        uint count = 256;
        nutty::HostBuffer<float3> hBuffer(count);

        NormalizedUniformValueGenerator rg(3 * count, m_seed, m_strength);
        
        rg.CreateRandomValues((float*)hBuffer.Begin()());

        m_randomDirections.Resize(256);
        nutty::Copy(m_randomDirections.Begin(), hBuffer.Begin(), hBuffer.Size());
    }

    void Turbulence::VUpdate(IParticleSystem* sys, IParticleEmitter* emitter, float time, float dt)
    {
        uint N = emitter->VGetParticleCount();
        uint grid = nutty::cuda::GetCudaGrid(N, CUDA_GROUP_SIZE);
        m_kernel.SetDimension(grid, CUDA_GROUP_SIZE);
        
        void* ptr = emitter->VGetParticleArray();
        void* accel = emitter->VGetAccelerationArray();
        m_kernel.SetRawKernelArg(0, &ptr);
        m_kernel.SetRawKernelArg(1, &accel);
        m_kernel.SetKernelArg(2, m_randomDirections);
        m_kernel.SetKernelArg(3, m_randomDirections.Size());
        m_kernel.SetKernelArg(4, time);
        m_kernel.SetKernelArg(5, N);
        m_kernel.Call();
    }

    uint Turbulence::VGetByteCount(void) const
    {
        return (uint)m_randomDirections.Size() * sizeof(float3);
    }
// 
//     typedef float* (*PROC)(uint,uint,uint,float);
// 
    GradientField::GradientField(void) : m_pArray(NULL)
    {
        float scale = 0.1f;

        //m_array = NULL;

        m_positionNscale.x = m_positionNscale.y = m_positionNscale.z = 0, m_positionNscale.w = scale;

        //CreateModActor(util::Vec3(0,0,0), "GradientField", scale);

        //warming up: todo...
        std::shared_ptr<IResHandle> handle = CmGetApp()->VGetCache()->VGetHandle(CMResource("kernel/Particles.ptx"));

        ADD_EVENT_LISTENER(this, &GradientField::OnResourceChangedDelegate, CM_EVENT_RESOURCE_CHANGED);
    }

    void GradientField::OnResourceChangedDelegate(IEventPtr event)
    {
        std::shared_ptr<ResourceChangedEvent> ce = std::static_pointer_cast<ResourceChangedEvent>(event);
        if(ce->m_resource == std::string("kernel/particles.ptx"))
        {
            VOnRestore(NULL, NULL);
        }
    }

    void GradientField::VOnRestore(IParticleSystem* sys, IParticleEmitter* emitter)
    {
        m_module.Create(KERNEL_PTX);
        m_kernel.Create(m_module.GetFunction("_computeGradientField"));

        uint w = 64;
        uint h = 64;
        uint d = 64;
        uint size = 4;

//         chimera::DLL dll("../../ParticleData/ParticleData/x64/Debug/ParticleData.dll");
//         float* data = dll.GetFunction<PROC>("CreateData")(w,h,d,1);
        //FLOAT* data = gradientpremades::RandomStuff2(w, h, d, 0.1f); //2, 16);//
        nutty::HostBuffer<float4> data(w * h * d);
        float4* ptr = data.Begin()();

        gradientpremades::GenerateNoiseValues(w, h, d, 2, 16, (float*)ptr);
        //gradientpremades::RandomStuff3(w, h, d, 1, (float*)ptr); //2, 16);//
        std::stringstream ss;
        ss << "gradientTexture_";
        ss << m_actorId;

        if(m_pArray)
        {
            cuArrayDestroy(m_pArray);
            m_pArray = NULL;
        }
        CUDA_ARRAY3D_DESCRIPTOR desc;
        ZeroMemory(&desc, sizeof(CUDA_ARRAY3D_DESCRIPTOR));
        desc.Depth = d;
        desc.Width = w;
        desc.Height = h;
        desc.NumChannels = 4;
        desc.Format = CU_AD_FORMAT_FLOAT;
        desc.Flags = 0;

        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuArray3DCreate(&m_pArray, &desc));
        
        CUDA_MEMCPY3D cpp;
        ZeroMemory(&cpp, sizeof(CUDA_MEMCPY3D));

        cpp.WidthInBytes = w * desc.NumChannels * sizeof(float);
        cpp.Depth = d;
        cpp.Height = h;
        cpp.srcHeight = h;
        cpp.srcPitch = cpp.WidthInBytes;
        cpp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        cpp.srcMemoryType = CU_MEMORYTYPE_HOST;
        cpp.dstArray = m_pArray;
        cpp.srcHost = (void*)ptr;

        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuMemcpy3D(&cpp));

        nutty::cuTexRef ref = m_module.GetTexRef("ct_gradientTexture");

        ref.SetFlags(CU_TRSF_NORMALIZED_COORDINATES);
        ref.SetFilterMode(CU_TR_FILTER_MODE_LINEAR);
        ref.SetAddressMode(CU_TR_ADDRESS_MODE_MIRROR, 0);
        ref.SetFormat(CU_AD_FORMAT_FLOAT, 4);
        ref.BindToArray(m_pArray);
        
        //m_array = sys->GetCuda()->CreateArray(ss.str(), w, h, d, cudah::eRGBA, data);


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
        //sys->GetCuda()->BindArrayToTexture("ct_gradientTexture", m_array, CU_TR_FILTER_MODE_LINEAR, CU_TR_ADDRESS_MODE_MIRROR, 0);

    }

    uint GradientField::VGetByteCount(void) const
    {
        return 64 * 64 * 64 * 16;
    }

    void GradientField::VUpdate(IParticleSystem* sys, IParticleEmitter* emitter, float time, float dt)
    {
        void* position = emitter->VGetParticleArray();
        void* velos = emitter->VGetVelocitiesArray();
        m_kernel.SetRawKernelArg(0, &position);
        m_kernel.SetRawKernelArg(1, &velos);
        float4 translation;
        ZeroMemory(&translation, sizeof(float4));
        m_kernel.SetKernelArg(2, translation);

        uint N = emitter->VGetParticleCount();
        m_kernel.SetKernelArg(3, N);

        uint grid = nutty::cuda::GetCudaGrid(N, CUDA_GROUP_SIZE);
        m_kernel.SetDimension(grid, CUDA_GROUP_SIZE);

        m_kernel.Call();
    }

    /*void GradientField::VOnActorMoved(std::shared_ptr<chimera::Actor> actor)
    {
        std::shared_ptr<chimera::TransformComponent> cmp = actor->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock();
        m_positionNscale.x = cmp->GetTransformation()->GetTranslation().x;
        m_positionNscale.y = cmp->GetTransformation()->GetTranslation().y;
        m_positionNscale.z = cmp->GetTransformation()->GetTranslation().z;
        m_positionNscale.w = cmp->GetTransformation()->GetScale().x;
    }*/

    GradientField::~GradientField(void)
    {
        REMOVE_EVENT_LISTENER(this, &GradientField::OnResourceChangedDelegate, CM_EVENT_RESOURCE_CHANGED);
        if(m_pArray)
        {
            cuArrayDestroy(m_pArray);
            m_pArray = NULL;
        }
    }
// 
//     GravityField::GravityField(const util::Vec3& position, const float range, const float scale, GravityPolarization pole)
//     {
//         m_posistionNrange.x = position.x;
//         m_posistionNrange.y = position.y;
//         m_posistionNrange.z = position.z;
//         m_posistionNrange.w = range;
//         m_pole = pole;
//         m_scale = scale;
//         CreateModActor(position, "GravityField", range);
//     }
// 
//     void GravityField::VOnRestore(ParticleSystem* sys)
//     {
//         m_kernel = sys->GetCuda()->GetKernel("_computeGravityField");
//     }
// 
//     void GravityField::VUpdate(ParticleSystem* sys, float time, float dt)
//     {
//         //computeGravityField((float4*)sys->GetParticles()->ptr, (float3*)sys->GetVelocities()->ptr, m_posistionNrange, m_pole == eRepel, m_scale, sys->GetParticlesCount(), sys->GetLocalWorkSize());
//         int threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
//         m_kernel->m_blockDim.x = sys->GetLocalWorkSize();
//         m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;
// 
//         int i = m_pole == eRepel;
// 
//         void *args[] = 
//         { 
//             &sys->GetParticles()->ptr, &sys->GetVelocities()->ptr, &m_posistionNrange, &i, &m_scale
//         };
// 
//         m_kernel->m_ppArgs = args;
// 
//         sys->GetCuda()->CallKernel(m_kernel);
//     }
// 
//     void GravityField::VOnActorMoved(std::shared_ptr<chimera::Actor> actor)
//     {
//         std::shared_ptr<chimera::TransformComponent> cmp = actor->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock();
//         m_posistionNrange.x = cmp->GetTransformation()->GetTranslation().x;
//         m_posistionNrange.y = cmp->GetTransformation()->GetTranslation().y;
//         m_posistionNrange.z = cmp->GetTransformation()->GetTranslation().z;
//         m_posistionNrange.w = cmp->GetTransformation()->GetScale().x;
//     }
// 
//     void VelocityDamper::VOnRestore(ParticleSystem* sys)
//     {
//         m_kernel = sys->GetCuda()->GetKernel("_computeVelocityDamping");
//     }
// 
//     void VelocityDamper::VUpdate(ParticleSystem* sys, float time, float dt)
//     {
//         //computeVelocityDamping((float4*)sys->GetParticles()->ptr, (float3*)sys->GetVelocities()->ptr, m_dampValue,  sys->GetParticlesCount(), sys->GetLocalWorkSize());
//         int threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
//         m_kernel->m_blockDim.x = sys->GetLocalWorkSize();
//         m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;
// 
//         void *args[] = 
//         { 
//             &sys->GetParticles()->ptr, &sys->GetVelocities()->ptr, &m_dampValue
//         };
// 
//         m_kernel->m_ppArgs = args;
// 
//         sys->GetCuda()->CallKernel(m_kernel);
//     }
// 
//     void Plane::VOnRestore(ParticleSystem* sys)
//     {
//         m_kernel = sys->GetCuda()->GetKernel("_computePlane");
//     }
// 
//     void Plane::VUpdate(ParticleSystem* sys, float time, float dt)
//     {
//         int threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
//         m_kernel->m_blockDim.x = sys->GetLocalWorkSize();
//         m_kernel->m_gridDim.x = threads / m_kernel->m_blockDim.x;
// 
//         DirectX::XMFLOAT4 n;
//         n.x = m_plane.GetNormal().m_v.x;
//         n.y = m_plane.GetNormal().m_v.y;
//         n.z = m_plane.GetNormal().m_v.z;
//         n.w = m_plane.GetRadius();
// 
//         void *args[] = 
//         { 
//             &n, &sys->GetVelocities()->ptr, &sys->GetParticles()->ptr,
//         };
// 
//         m_kernel->m_ppArgs = args;
// 
//         sys->GetCuda()->CallKernel(m_kernel);
//     }
// 
//     BoundingBox::BoundingBox(void)
//     {
//         m_pData[0] = NULL;
//         m_pData[1] = NULL;
//     }
// 
//     BoundingBox::~BoundingBox(void)
//     {
//         SAFE_ARRAY_DELETE(m_pData[0]);
//         SAFE_ARRAY_DELETE(m_pData[1]);
//     }
// 
//     void BoundingBox::VOnRestore(ParticleSystem* sys)
//     {
//         SAFE_ARRAY_DELETE(m_pData[0]);
//         SAFE_ARRAY_DELETE(m_pData[1]);
//         m_pData[0] = new float[3 * 1024];
//         m_pData[1] = new float[3 * 1024];
//         m_kernel = sys->GetCuda()->GetKernel("_reduce_max4");
//         m_second = sys->GetCuda()->GetKernel("_reduce_min4");
//         m_min = sys->GetCuda()->CreateBuffer(std::string("__min"), 1024 * 3 * sizeof(float), 3 * sizeof(float));
//         m_max = sys->GetCuda()->CreateBuffer(std::string("__max"), 1024 * 3 * sizeof(float), 3 * sizeof(float));
//     }
// 
//     void _GetAABB(float3* data[2], uint l, util::AxisAlignedBB& aabb)
//     {
//         aabb.Clear();
//         for(uint i = 0; i < l; ++i)
//         {
//             float3 m0 = data[0][i];
//             float3 m1 = data[1][i];
//             aabb.AddPoint(util::Vec3(m0.x, m0.y, m0.z));
//             aabb.AddPoint(util::Vec3(m1.x, m1.y, m1.z));
//         }
//         aabb.Construct();
//     }
// 
//     void BoundingBox::VUpdate(ParticleSystem* sys, float time, float dt)
//     {
//         int threads = cudahu::GetThreadCount(sys->GetParticlesCount(), sys->GetLocalWorkSize());
//         m_kernel->m_blockDim.x = 512;//sys->GetLocalWorkSize();
//         m_kernel->m_gridDim.x = (threads/2) / m_kernel->m_blockDim.x;
//         m_kernel->m_shrdMemBytes = 3 * (m_kernel->m_blockDim.x) * sizeof(float);
//         void *args[] = 
//         { 
//             &sys->GetParticles()->ptr, &m_max->ptr
//         };
// 
//         m_kernel->m_ppArgs = args;
// 
//         sys->GetCuda()->CallKernel(m_kernel);
// 
//         m_second->m_blockDim.x = m_kernel->m_blockDim.x;
//         m_second->m_gridDim.x = m_kernel->m_gridDim.x;
//         m_second->m_shrdMemBytes = m_kernel->m_shrdMemBytes;
// 
//         void* args2[] = 
//         { 
//             &sys->GetParticles()->ptr, &m_min->ptr
//         };
// 
//         m_second->m_ppArgs = args2;
// 
//         sys->GetCuda()->CallKernel(m_second);
//         
//         sys->GetCuda()->ReadBuffer(m_max, m_pData[0]);
// 
//         sys->GetCuda()->ReadBuffer(m_min, m_pData[1]);
// 
//         _GetAABB((float3**)m_pData, 1024, m_aabb);
// 
//         sys->SetAxisAlignedBB(m_aabb);
//     }
// 
//     uint BoundingBox::VGetByteCount(void)
//     {
//         return m_min->GetByteSize() + m_max->GetByteSize();
//     }
}