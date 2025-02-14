import streamlit as st
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate("config/config.yaml")

try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state["authentication_status"]:
    authenticator.logout("Terminar sessão", "main")
    st.title("Avaliação de trocadilhos")
    st.markdown(f'''
    Bem-vindo(a) *{st.session_state['name']}*!

    Obrigado por aceitar participar dessa avaliação.
    ''')
elif st.session_state["authentication_status"] == False:
    st.error("Usuário/senha incorreta")
