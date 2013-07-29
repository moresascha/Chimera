#include "Maze.h"
#include "Components.h"
#include "GameLogic.h"
#include "ActorFactory.h"
#include "GameApp.h"
#include "Packman.h"

namespace packman
{
    FLOAT rrand()
    {
        return rand() / (FLOAT) RAND_MAX;
    }

    struct Vec2
    {
        INT x;
        INT z;
        Vec2(VOID) : x(0), z(0)
        {

        }

        Vec2(INT x, INT z) : x(x), z(z)
        {

        }

    };

    struct Wall
    {
        Vec2 start;
        Vec2 end;

        Wall(Vec2 start, Vec2 end) : start(start), end(end)
        {

        }

        Wall(CONST Wall& wall)
        {
            this->start = wall.start;
            this->end = wall.end;
        }

        Wall(VOID)
        {

        }
    };

    struct WallSet
    {
        Vec2 doors[4];
        Wall wx;
        Wall wz;

        WallSet(VOID)
        {

        }

        WallSet(Wall& wx, Wall& wz) : wx(wx), wz(wz)
        {
            doors[0] = Vec2(1 + wx.start.x + (int)(rrand() * (wz.start.x - wx.start.x)), wx.start.z);
            doors[1] = Vec2(wz.start.x, 1 + wz.start.z + (int)(rrand() * (wx.start.z - wz.start.z)));
            doors[2] = Vec2(1 + wz.start.x + (int)(rrand() * (wx.end.x - wz.start.x)), wx.start.z);
            doors[3] = Vec2(wz.start.x, 1 + wx.start.z + (int)(rrand() * (wz.end.z - wx.start.z)));
        }

        BOOL IsDoor(INT x, INT z)
        {
            for(INT i = 0; i < 4; ++i)
            {
                if(doors[i].x == x && doors[i].z == z)
                {
                    return TRUE;
                }
            }
            return FALSE;
        }

        BOOL IsWall(INT x, INT z)
        {
            BOOL b0 = wx.start.z == z && wx.end.z == z;
            BOOL b1 = (wx.start.x <= x && wx.end.x >= x);

            BOOL b2 = (wz.start.x == x && wz.end.x == x);
            BOOL b3 = (wz.start.z <= z && wz.end.z >= z);

            return !IsDoor(x, z) && (b0 && b1 || b2 && b3);
        }
    };

    struct Chamber
    {
        Vec2 m_min;
        Vec2 m_max;
        WallSet ws;

        Chamber(VOID)
        {

        }

        Chamber(Vec2& min, Vec2& max) : m_min(min), m_max(max)
        {
            int sz = this->m_min.z + (int)(rrand() * (this->m_max.z - this->m_min.z));
            sz = sz == this->m_max.z ? this->m_max.z - 1 : (sz == this->m_min.z ? sz + 1 : sz);
            Wall wx = Wall(Vec2(this->m_min.x, sz), Vec2(this->m_max.x, sz));

            int sx = this->m_min.x + (int)(rrand() * (this->m_max.x - this->m_min.x));
            sx = sx == this->m_max.x ? this->m_max.x - 1 : (sx == this->m_min.x ? sx + 1 : sx);
            Wall wz = Wall(Vec2(sx, this->m_min.z), Vec2(sx, this->m_max.z));

            ws = WallSet(wx, wz);
        }

        Chamber GetChild(INT index)
        {
            switch(index)
            {
            case 0 :
                {
                    return Chamber(Vec2(m_min.x + 1, m_min.z + 1), Vec2(ws.wz.end.x - 1, ws.wx.end.z - 1));
                } break;
            case 1 :
                {
                    return Chamber(Vec2(ws.wz.end.x + 1, m_min.z + 1), Vec2(ws.wx.end.x - 1, ws.wx.end.z - 1));
                } break;
            case 2 :
                {
                    return Chamber(Vec2(ws.wz.end.x + 1, ws.wx.end.z + 1), Vec2(m_max.x, m_max.z));
                } break;
            case 3 :
                {
                    return Chamber(Vec2(m_min.x + 1, ws.wx.end.z + 1), Vec2(ws.wz.end.x - 1, ws.wz.end.z - 1));
                } break;

            default : return Chamber();
            }
        }

        BOOL IsValid(VOID)
        {
            INT x = m_max.x - m_min.x;
            INT z = m_max.z - m_min.z;
            return (x > 3 && z > 3);
        }

    };

    std::shared_ptr<tbd::Actor> CreateEnemy(CONST util::Vec3& pos,  Maze* level)
    {
        tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

        tbd::RenderComponent* renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
        renderComp->m_meshFile = "sphere.obj";

        tbd::PhysicComponent* physicComponent = desc->AddComponent<tbd::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = "kinematic";
        physicComponent->m_shapeType = "sphere";
        physicComponent->m_radius = 1;

        packman::AIComponent* aiComp = desc->AddComponent<packman::AIComponent>(packman::AIComponent::COMPONENT_ID);

        return level->VAddActor(desc);
    }

    VOID CreateMaze(Chamber& c, std::list<Chamber>& chambers, INT* step)
    {
        (*step)++;
        if(!c.IsValid())
        {
            return;
        }
        chambers.push_back(c);
        for(INT i = 0; i < 4; ++i)
        {
            CreateMaze(c.GetChild(i), chambers, step);
        }
    }

    Maze::Maze(INT size, INT enemies, tbd::ActorFactory* factory) : BaseLevel("test", factory), m_size(size), m_step(0), m_enemies(enemies)
    {
    }

    BOOL Maze::VLoad(BOOL block)
    {

        tbd::CreateStaticPlane(this);

        srand(100);
        std::list<Chamber> list;
        CONST INT size = (INT)m_size;
        Chamber c(Vec2(0, 0), Vec2(size, size));
        CreateMaze(c, list, &m_step);

        m_vals = new INT*[size+1];

        for(INT i = 0; i < size + 1; ++i)
        {
            m_vals[i] = new INT[size + 1];
        }

        for(INT i = 0; i < size+1; ++i)
        {
            for(INT j = 0; j < size+1; ++j)
            {
                m_vals[i][j] = FALSE;
            }
        }

        TBD_FOR(list)
        {
            for(INT i = 0; i < size+1; ++i)
            {
                for(INT j = 0; j < size+1; ++j)
                {
                    m_vals[i][j] |= it->ws.IsWall(i, j);
                }
            }
        }

        //CreateCube(util::Vec3(-2,0,0), this);

        tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
        //comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

        tbd::RenderComponent* renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
        renderComp->m_meshFile = "box.obj";

        tbd::PhysicComponent* physicComponent = desc->AddComponent<tbd::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = "static";
        physicComponent->m_shapeType = "box";
        physicComponent->m_radius = 1;

        //renderComp->m_instances.push_back(util::Vec3(0,1,0));

        FLOAT s = 2;
        for(int i = 0; i < size + 1; ++i)
        {
            //std::stringstream ss;
            for(int j = 0; j < size + 1; ++j)
            {
                if(m_vals[i][j])
                {

                    //CreateCube(util::Vec3(-size + s * i, 1, -size + s * j), this);
                    //CreateCube(util::Vec3(-size + s * i, 3, -size + s * j), this);
                    //CreateCube(util::Vec3(-size + s * i, 5, -size + s * j), this);
                    renderComp->m_instances.push_back(util::Vec3(-size + s * i, 1, -size + s * j));
                    //renderComp->m_instances.push_back(util::Vec3(-size + s * i, 3, -size + s * j));
                    //renderComp->m_instances.push_back(util::Vec3(-size + s * i, 5, -size + s * j));
                }
                //ss << (vals[i][j] ? " *" : " -");
            }
            //DEBUG_OUT(ss.str());
        } 

        VAddActor(desc);

        for(INT i = 0; i < m_enemies; ++i)
        {
            INT x;
            INT z;
            do 
            {
                x = -m_size + 2 * (INT)(rrand() * m_size);
                z = -m_size + 2 * (INT)(rrand() * m_size);
            } while (IsWall((FLOAT)x, (FLOAT)z));

            std::shared_ptr<tbd::Actor> actor = CreateEnemy(util::Vec3((FLOAT)x, 1, (FLOAT)z), this);
        }

        desc = m_pActorFactory->CreateActorDescription();
        comp = desc->AddComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID);
        comp->GetTransformation()->SetTranslate(0,0,0);

        renderComp = desc->AddComponent<tbd::RenderComponent>(tbd::RenderComponent::COMPONENT_ID);
        renderComp->m_type = "skydome";
        renderComp->m_info = "skydome3.jpg";
        VAddActor(desc);

        return TRUE;
    }

    BOOL Maze::VSave(LPCSTR file /* = NULL */)
    {
        return TRUE;
    }

    FLOAT Maze::VGetLoadingProgress(VOID)
    {
        //INT i = log(m_size / 3.0) / log(2.0);
        return BaseLevel::VGetLoadingProgress();// * 0.5f + 0.5f * (m_step / (2 << i));
    }

    BOOL Maze::IsWall(FLOAT x, FLOAT z)
    {
        INT ix = (INT)(((x + (FLOAT)m_size) / 2.0f));
        INT iy = (INT)(((z + (FLOAT)m_size) / 2.0f));
        if(ix < 0 || ix > m_size || iy < 0 || iy > m_size)
        {
            return FALSE;
        }
        return m_vals[ix][iy];
    }

    Maze::~Maze(VOID)
    {
        for(INT i = 0; i < m_size + 1; ++i)
        {
            SAFE_ARRAY_DELETE(m_vals[i]);
        }

        SAFE_ARRAY_DELETE(m_vals);
    }
}