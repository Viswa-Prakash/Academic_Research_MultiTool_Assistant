import streamlit as st
from langchain_core.messages import HumanMessage
from ReAct_Agent import research_agent

st.set_page_config(page_title="Academic Research Assistant", page_icon=":brain:")

st.title("Academic Research Multi Tool Assistant")

st.markdown("""
            Summarize the most cited papers in reinforcement learning from the last two years. 
            Also, translate the main findings into Spanish
            """)

with st.form("user_form"):
    user_query = st.text_area("Enter your query here :", height=60)
    submitted = st.form_submit_button("Ask Agent")

if submitted and user_query.strip():
    with st.spinner("Agent is analyzing..."):
        output = research_agent.invoke({"messages": [HumanMessage(content=user_query)]})
        # Show **only the last agent message** (the Final Answer)
        final_message = None
        for msg in reversed(output["messages"]):
            content = getattr(msg, "content", "")
            if "final answer" in content.lower():
                final_message = content
                break
        if not final_message:
            # fallback: just show last assistant/system message
            last = output["messages"][-1]
            final_message = getattr(last, "content", str(last))
        st.markdown("**Here’s a clear summary of your requests and answers:**\n\n" + final_message)
