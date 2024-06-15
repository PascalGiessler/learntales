import time
from typing import List

import streamlit as st
import markdown

from app.services.llm import OllamaChatModel, NvidiaFoundationChatModel, BaseChatModel
from app.services.storyteller import StoryTeller
from app.services.vision_model import NvidiaFoundationVisionModel
from loguru import logger

placeholder = """
Das Land wurde zentral vom ägyptischen König (Pharao) regiert, der als Sohn des Sonnengottes Re galt. Das Volk verehrte ihn als einen Vertreter des Göttlichen auf Erden und damit Inhaber eines göttlichen Amtes. Als Herrscher besaß er uneingeschränkte Machtbefugnisse. Er war der einzige Eigentümer von Grund und Boden mit allen darauf befindlichen Produkten, und er verfügte über Bodenschätze sowie die Beute aus Kriegszügen.[2]  In der Regel übte der König seine Herrschaft von der Thronbesteigung an bis zum Lebensende aus. Seine Nachfolge trat der älteste mit der Hauptgemahlin gezeugte Sohn an. König und Königsfamilie waren in einem eigenen Palast untergebracht, der sowohl öffentlich als auch privat genutzt wurde und sich zumeist in der Hauptstadt des Landes befand.  Der ägyptische König hatte für das absolute Wohl des Landes und für die Aufrechterhaltung der Weltordnung (Maat) zu sorgen. Er erließ alle Gesetze und Dekrete, überwachte Wirtschaft und Handel, besaß die Oberbefehlsgewalt über das Heer und bestimmte das Bauprogramm, insbesondere den Bau von Tempeln. Daneben ließ er notwendige Reformen durchführen, ernannte oberste Minister, die ihn bei der Ausübung seines Regierungsamtes unterstützten und verlieh das Ehrengold an seine Untergebenen für besondere Leistungen. Darüber hinaus sorgte er im ganzen Land für die Aufrechterhaltung der Tempelkulte, die von stellvertretenden Priestern durchgeführt wurden. Große Sorgfalt galt der Vorbereitung auf sein ewiges Leben. Mit der Anlage des Königsgrabes wurde meist schon während seines Regierungsantrittes begonnen.  Im 30. Regierungsjahr, und dann darauf folgend jedes weitere dritte Jahr, wurde das Sedfest gefeiert, das zur rituellen Erneuerung des Königtums diente. Weitere Rituale und Feste waren die Jagd auf Großwild und Löwen sowie das Vereinigungsfest, bei dem der König sich als Nachfolger des vermeintlich ersten Königs und Reichseinigers Menes feiern ließ.  Zu den typischen Insignien des Herrschers zählten die Doppelkrone, die Uräusschlange und der Zeremonialbart.
"""

intro_text = """
Imagine a world where learning is an adventure, where stories come alive and transform complex
information into enchanting experiences. Welcome to Learn Tales, the revolutionary software designed
to generate joyful, immersive stories that make learning an absolute delight for children"
"""

st.set_page_config(page_title="LearnTales", layout="wide")

st.markdown(
    """
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def init_models(api_key: str | None = None, debug: bool = False):
    logger.info("loading the models")
    logger.info(api_key)

    llm = NvidiaFoundationChatModel(api_key=api_key if api_key is not None else "")
    if debug:
        llm = OllamaChatModel()

    vision_model = NvidiaFoundationVisionModel(
        api_key=api_key if api_key is not None else ""
    )
    return llm, vision_model


def get_model_options(_llm: BaseChatModel, api_key: str | None) -> List[str]:
    logger.info("getting llm models")
    if _llm.is_api_key_needed() and api_key is None:
        return []

    models = list({llm[1] for llm in _llm.get_available_models()})
    return models


def reset():
    st.session_state.content = None
    st.session_state.story = None
    st.session_state["story_generation"] = False


def stream_data():
    for word in (
        "Imagine a world where learning is an adventure, where stories come alive and transform complex "
        "information into enchanting experiences. Welcome to Learn Tales, the revolutionary software designed"
        " to generate joyful, immersive stories that make learning an absolute delight for children"
    ).split(" "):
        yield word + " "
        time.sleep(0.02)


content = st.session_state.content if "content" in st.session_state else None


def tell_me():
    if "api_key" not in st.session_state:
        st.error("You need to provide a valid NIM API Key", icon="⚠️")
        return

    st.session_state["story_generation"] = True
    content = st.session_state.content
    model_name = st.session_state.model
    language = st.session_state.language
    from_year, to_year = st.session_state.target_age

    story_teller = StoryTeller(visionModel=visionModel, llm=llm)
    original_content = content
    story = story_teller.tell(
        original_content=original_content,
        model_name=model_name,
        audience="children",
        from_year=from_year,
        to_year=to_year,
        language=language,
    )
    st.session_state.story = story
    st.session_state.content = None
    st.balloons()


if "story_generation" in st.session_state and st.session_state["story_generation"]:
    logger.info("Generating story")
else:
    st.title("LearnTales")
    st.write(intro_text)
    st.text_area(
        label="Let me know your information content for story generation",
        key="content",
        placeholder=placeholder,
        height=500,
    )
    st.button(
        label="Tell me a story",
        on_click=tell_me,
        disabled=(
            len(st.session_state.content) == 0
            if "content" in st.session_state and st.session_state.content is not None
            else True
        ),
    )

with st.sidebar:
    api_key = st.session_state["api_key"] if "api_key" in st.session_state else None
    llm, visionModel = init_models(
        api_key=api_key,
        debug=st.session_state["debug"] if "debug" in st.session_state else False,
    )
    model_options = get_model_options(_llm=llm, api_key=api_key)
    st.image("./assets/cover.jpeg", width=150)
    st.checkbox(label="Debug with (Ollama)", key="debug")
    st.markdown(
        "Idea of generating small engaging stories with informational content for micro learning"
    )
    st.title("Settings")
    if llm.is_api_key_needed():
        api_key = st.text_input("API Key", key="api_key", type="password")
    model = st.selectbox("LLM", options=model_options, key="model")
    target_language = st.selectbox(
        "Language", options=["german", "english"], key="language"
    )
    from_age, to_age = st.select_slider(
        "Age",
        key="target_age",
        options=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        value=(5, 7),
    )
    st.markdown("Powered by Nvidia")

if "story" in st.session_state and st.session_state.story is not None:
    st.markdown(markdown.markdown(st.session_state.story), unsafe_allow_html=True)
    st.button("Tell me a new story", on_click=reset)
