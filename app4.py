import streamlit as st
import os
from PIL import Image
from rag_chatbot2 import rag_chatbot
import re

st.set_page_config(page_title="RAG Chatbot with Image Context", layout="wide")
# Check if a file is a valid image
def is_image_file(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧠 RAG Chatbot with Image Context")

# User input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Send")

# Handle query submission
if submitted and user_input.strip():
    st.session_state.chat_history.append(("You", user_input))

    # Call backend RAG chatbot
    response, elapsed_time, references = rag_chatbot(user_input)

    # Append bot response to chat history
    st.session_state.chat_history.append(("Bot", response, elapsed_time, references))

# Display previous conversation
if st.session_state.chat_history:
    for item in st.session_state.chat_history:
        if item[0] == "You":
            st.markdown(f"**🧑 You:** {item[1]}")
        else:
            response, elapsed_time, references = item[1], item[2], item[3]

            # Separate main answer from references if they exist
            if "References:" in response:
                main_answer, _ = response.split("References:", 1)
            else:
                main_answer = response

            st.markdown(f"**🤖 Bot:** {main_answer.strip()}")

            # Show reference section
            if references:
                with st.expander("📁 References"):
                    for ref in references:
                        st.markdown(f"### 🔹 {ref['label']}")
                        st.markdown(f"**📄 Source Path:** `{ref['path']}`")

                        if ref.get("images"):
                            st.markdown("**🖼️ Images Found:**")
                            for img_path in ref["images"]:
                                if not os.path.exists(img_path):
                                    st.markdown(f"- ❌ Image not found: `{img_path}`")
                                    continue
                                if img_path.lower().endswith('.svg'):
                                    try:
                                        with open(img_path, 'r', encoding='utf-8') as f:
                                            svg_content = f.read()
                                        st.components.v1.html(svg_content, height=300, scrolling=False)
                                    except Exception as e:
                                        st.markdown(f"- ❌ Could not display SVG: `{img_path}` Error: {e}")

                                elif img_path.lower().endswith(".svg.js"):
                                    try:
                                        with open(img_path, 'r', encoding='utf-8') as f:
                                            js_content = f.read()

                                        # Try to extract SVG inside quotes
                                        # match = re.search(r'["\'](<svg.*?</svg>)["\']', js_content, re.DOTALL)
                                        match = re.search(r'<svg\b[^>]*>(.*?)</svg>', js_content, re.DOTALL | re.IGNORECASE)
                                        if match:
                                            svg_raw = match.group(1)
                                            st.components.v1.html(svg_raw, height=300, scrolling=False)
                                        else:
                                            st.markdown(f"- ⚠️ Could not extract embedded SVG from: `{img_path}`")
                                    except Exception as e:
                                        st.markdown(f"- ❌ Error reading `.svg.js` file: `{img_path}` Error: {e}")
                                elif is_image_file(img_path):
                                    st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                                else:
                                    st.markdown(f"- ❌ Unsupported or corrupted image: `{img_path}`")
            st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
            st.markdown("---")
else:
    st.info("Ask something to get started!")
