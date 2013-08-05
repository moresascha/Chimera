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
namespace tbd
{
    class TransformComponent;
    namespace gui
    {
        class InformationWindow : public tbd::gui::GuiTextComponent
        {
            tbd::gui::Histogram m_eventHisto;
        public:
            InformationWindow(VOID) : m_eventHisto(100)
            {
                SetAlignment(tbd::gui::eRight);
                SetTextColor(util::Vec4(0.75f, 0.75f, 0.75f, 0));
            }

            VOID VDraw(VOID)
            {
                GuiTextComponent::VDraw();
                m_eventHisto.VDraw();
            }

            BOOL VOnRestore(VOID)
            {
                Dimension dim;
                dim.x = 0;//(INT)(0.4 * d3d::g_width);
                dim.y = 0;
                dim.w = (INT)(0.25 * app::g_pApp->GetWindowWidth());
                dim.h = 70;
                m_eventHisto.VSetDimension(dim);
                m_eventHisto.VSetAlpha(0.5f);
                m_eventHisto.VOnRestore();
                return GuiTextComponent::VOnRestore();
            }

            VOID VUpdate(ULONG millis)
            {
                ClearText();
                tbd::TransformComponent* tc = app::g_pApp->GetHumanView()->GetTarget()->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock().get();
                std::stringstream ss;
                ss << "FPS:";
                ss << app::g_pApp->GetRenderingTimer()->GetFPS();
                AppendText(ss.str());
                ss.str("");

                ss << "Ticks:";
                ss << app::g_pApp->GetUpdateTimer()->GetFPS();
                AppendText(ss.str());
                ss.str("");

                ss << "View: ";
                ss << app::g_pApp->GetHumanView()->GetTarget()->GetName();
                ss << " (";
                ss << tc->GetTransformation()->GetTranslation().x;
                ss << ", ";
                ss << tc->GetTransformation()->GetTranslation().y;
                ss << ", ";
                ss << tc->GetTransformation()->GetTranslation().z;
                ss << ")";
                AppendText(ss.str());
                ss.str("");

                ss << "CacheLoad=";
                ss << app::g_pApp->GetCache()->GetWorkload();
                ss << "%";
                AppendText(ss.str());
                ss.str("");

                ss << "VRamLoad=";
                ss << app::g_pApp->GetHumanView()->GetVRamManager()->GetWorkload();
                ss << "%";
                AppendText(ss.str());
                ss.str("");

                ss << "LastEventsFired=";
                ss << app::g_pApp->GetEventMgr()->LastEventsFired();
                AppendText(ss.str());
                ss.str("");

                m_eventHisto.AddValue(app::g_pApp->GetEventMgr()->LastEventsFired());

                if(app::g_pApp->GetHumanView()->GetPicker()->VHasPicked())
                {
                    ss << "Actor: ";
                    ss << app::g_pApp->GetHumanView()->GetPicker()->VPick();
                    AppendText(ss.str());

                    ss.str("");
                    AppendText("Components:");
                    std::shared_ptr<tbd::Actor> picked = app::g_pApp->GetLogic()->VFindActor(app::g_pApp->GetHumanView()->GetPicker()->VPick());
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

