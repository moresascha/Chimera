#pragma once
#include "stdafx.h"
#include "GuiComponent.h"
#include "GameApp.h"
#include "Timer.h"
#include "Resources.h"
#include "VRamManager.h"
#include "GameView.h"
#include "EventManager.h"
#include "Picker.h"
namespace chimera
{
    class TransformComponent;
    namespace gui
    {
        class InformationWindow : public chimera::gui::GuiTextComponent
        {
            chimera::gui::Histogram m_eventHisto;
        public:
            InformationWindow(VOID) : m_eventHisto(100)
            {
                SetAlignment(chimera::gui::eRight);
                SetTextColor(util::Vec4(0.75f, 0.75f, 0.75f, 0));
            }

            VOID VDraw(VOID)
            {
                GuiTextComponent::VDraw();
                m_eventHisto.VDraw();
            }

            BOOL VOnRestore(VOID)
            {
                CMDimension dim;
                dim.x = 0;//(INT)(0.4 * d3d::g_width);
                dim.y = 0;
                dim.w = (INT)(0.25 * chimera::g_pApp->GetWindowWidth());
                dim.h = 70;
                m_eventHisto.VSetDimension(dim);
                m_eventHisto.VSetAlpha(0.5f);
                m_eventHisto.VOnRestore();
                return GuiTextComponent::VOnRestore();
            }

            VOID VUpdate(ULONG millis)
            {
                ClearText();
                chimera::TransformComponent* tc = chimera::g_pApp->GetHumanView()->GetTarget()->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock().get();
                std::stringstream ss;
                ss << "FPS:";
                ss << chimera::g_pApp->GetRenderingTimer()->GetFPS();
                AppendText(ss.str());
                ss.str("");

                ss << "Ticks:";
                ss << chimera::g_pApp->GetUpdateTimer()->GetFPS();
                AppendText(ss.str());
                ss.str("");

                ss << chimera::GetAdapterName();
                AppendText(ss.str());
                ss.str("");

                ss << "Position: ";
                ss << chimera::g_pApp->GetHumanView()->GetTarget()->GetName();
                ss << " (";
                ss << tc->GetTransformation()->GetTranslation().x;
                ss << ", ";
                ss << tc->GetTransformation()->GetTranslation().y;
                ss << ", ";
                ss << tc->GetTransformation()->GetTranslation().z;
                ss << ")";
                AppendText(ss.str());
                ss.str("");

                ss << "CacheLoad: ";
                ss << chimera::g_pApp->GetCache()->GetWorkload();
                ss << "%";
                AppendText(ss.str());
                ss.str("");

                ss << "VRamLoad: ";
                ss << chimera::g_pApp->GetHumanView()->GetVRamManager()->GetWorkload();
                ss << "%";
                AppendText(ss.str());
                ss.str("");

                ss << "LastEventsFired ";
                ss << chimera::g_pApp->GetEventMgr()->LastEventsFired();
                AppendText(ss.str());
                ss.str("");

                m_eventHisto.AddValue(chimera::g_pApp->GetEventMgr()->LastEventsFired());

                if(chimera::g_pApp->GetHumanView()->GetPicker()->VHasPicked())
                {
                    ss << "Actor: ";
                    ss << chimera::g_pApp->GetHumanView()->GetPicker()->VPick();
                    AppendText(ss.str());

                    ss.str("");
                    AppendText("Components:");
                    std::shared_ptr<chimera::Actor> picked = chimera::g_pApp->GetLogic()->VFindActor(chimera::g_pApp->GetHumanView()->GetPicker()->VPick());
                    if(picked)
                    {
                        TBD_FOR(picked->GetComponents())
                        {
                            AppendText(it->second->VGetName());
                        }
                    }
                }
            }

            ~InformationWindow(VOID)
            {
            }
        };
    }
}

