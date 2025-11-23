# app.py
import streamlit as st
from utils.config import OPEN_ROUTER_API
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI
from tools import faiss_tool, tivaly_tool
import time

st.set_page_config(
    page_title="Live Web-QA Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ---- CSS ----
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);
    }

    .chat-wrapper {
        width: 100%;
        display: flex;
        margin: 8px 0;
    }

    .right {
        justify-content: flex-end;
    }

    .left {
        justify-content: flex-start;
    }

    .bubble {
        padding: 10px 14px;
        border-radius: 14px;
        max-width: 80%;
        display: inline-block;
    }

    .user {
        background: linear-gradient(90deg, #87e0fd 0%, #53cbf1 100%);
        color: black;
    }

    .bot {
        background: #f1f5ff;
        color: black;
    }

    .meta {
        font-size: 11px;
        color: #6b7280;
        margin-bottom: 3px;
    }

    .header {
        display:flex;
        align-items:center;
        gap:12px;
    }

    .logo {
        width:52px;
        height:52px;
        border-radius:12px;
        background: linear-gradient(135deg,#7dd3fc,#60a5fa);
        display:flex;
        align-items:center;
        justify-content:center;
        color:white;
        font-weight:700;
        font-size:20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- HEADER ----
st.markdown(
    """
    <div class='header'>
        <div class='logo'>QA</div>
        <div>
            <h2 style='margin:0'>Live Web-QA Chatbot</h2>
            <div style='color:#475569'>Search-first RAG: FAISS local cache → Tivaly fallback</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---- Initialize agent ----
if "agent" not in st.session_state:
    with st.spinner("Initializing agent..."):
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPEN_ROUTER_API,
            model="gpt-4o-mini",
            temperature=0.25,
            max_tokens=2048
        ).with_config({"verbose": False})

        tools = [faiss_tool, tivaly_tool]

        st.session_state.agent = initialize_agent(
            tools,
            llm,
            agent="zero-shot-react-description",
            verbose=False
        )

# ---- Chat history ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---- Message Renderer ----
def render_message(role, text):
    is_user = (role == "user")

    align_class = "right" if is_user else "left"
    bubble_class = "user" if is_user else "bot"
    meta = "You" if is_user else "Bot"

    st.markdown(
        f"""
        <div class="chat-wrapper {align_class}">
            <div>
                <div class="meta">{meta}</div>
                <div class="bubble {bubble_class}">{text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---- Display chat ----
for role, msg in st.session_state.chat_history:
    render_message(role, msg)


# ---- Input ----
with st.form("query_form", clear_on_submit=True):
    user_input = st.text_input("Ask something...", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # store RAW TEXT (no DeltaGenerator)
    st.session_state.chat_history.append(("user", user_input))

    # Temp placeholder while bot thinks
    placeholder = st.empty()
    placeholder.markdown(
        """
        <div class="chat-wrapper left">
            <div>
                <div class="meta">Bot</div>
                <div class="bubble bot">Thinking…</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        with st.spinner("Retrieving and synthesizing answer..."):
            answer = st.session_state.agent.run(user_input)

        st.session_state.chat_history.append(("bot", answer))

    except Exception as e:
        st.session_state.chat_history.append(("bot", f"Error: {e}"))

    placeholder.empty()
    st.experimental_rerun()


# ---- Footer ----
st.markdown(
    "<div style='text-align:center;margin-top:12px;color:#94a3b8;font-size:13px'>Deployed on Hugging Face Spaces / Run locally with `streamlit run app.py`</div>",
    unsafe_allow_html=True,
)
