import streamlit as st
import streamlit_authenticator as stauth

st.set_page_config(layout="wide")
authenticator = stauth.Authenticate("config/config.yaml")

try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state["authentication_status"]:
    authenticator.logout("Terminar sessão", "main")
    st.title("Avaliação de trocadilhos")
    st.markdown(f"""
    Bem-vindo(a) *{st.session_state["name"]}*!

    Obrigado por ter aceitado em participar.
    """)

    with st.expander("Avaliação de teste", expanded=True):
        st.write("Original: Qual é a sobremesa mais popular na Rússia? O Putin flan.")
        st.write("Editada: Qual é a sobremesa mais popular na Rússia? O pudim flan.")
        st.divider()
        st.write("Original: Um parto não costuma demorar muito tempo. Mas para as grávidas parece maternidade.")
        st.write("Editada: Um parto não costuma demorar muito tempo. Mas para as grávidas parece uma eternidade.")
        st.divider()
        st.write("Original: Qual cantora superou seu deficit de atencao? Rita Li-na")
        st.write("Editada: Qual cantora superou seu deficit de atencao? Ana Carolina")
elif st.session_state["authentication_status"] == False:
    st.error("Usuário/senha incorreta")
