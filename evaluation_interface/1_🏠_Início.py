import streamlit as st
import streamlit_authenticator as stauth
from utils import load_config, load_custom_style
from streamlit_sortables import sort_items

st.set_page_config(layout="wide")

authenticator = stauth.Authenticate("config/credentials.yaml")
try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state.authentication_status:
    authenticator.logout("Terminar sessão", "main")
    st.title("Avaliação de trocadilhos")
    st.markdown(f"""
    Bem-vindo(a) *{st.session_state.name}*!

    Obrigado por ter aceitado em participar desta avaliação. O objetivo é analisar a qualidade de piadas geradas automaticamente a partir de títulos de notícias.

    Essa avaliação esta estuturada da seguinte forma. Você receberá um máximo de 5 noticias, uma por vez, em uma página com o título no topo e dois rankings abaixo, cada ranking contém as mesmas 10 piadas. Para realizar a avaliação, você deverá reorganinzar as piadas de acordo com os critérios abaixo, sendo a posição de número 1 a mais adequada e a posição de número 10 a menos adequada.

    Os rankings devem seguir dois critérios diferentes:
    - Nível de humor (ranking da esquerda);
    - Relação com o título (ranking da direita).

    Aqui embaixo você verá um pequeno exemplo (com 3 piadas) para entender como é a interface de avaliação, pode brincar à vontade! Se tiver qualquer dúvida, pode vir falar comigo.
    """)

    with st.expander("Avaliação de teste", expanded=True):
        with st.container(border=True):
            st.markdown(f"<p style=\"font-size:30px\">ChatGPT é visto dirigindo Uber após chegada de IA chinesa</p>",
                        unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large", border=True)
        with col1:
            st.markdown("Qual piada **tem mais piada**?")
            sort_items(["Piada 1", "Piada 2", "Piada 3"], "",
                       custom_style=load_custom_style(),
                       direction="vertical",
                       key="test_fun_rank")
        with col2:
            st.markdown("Qual piada **mais se relaciona** com o título?")
            sort_items(["Piada 1", "Piada 2", "Piada 3"], "",
                       custom_style=load_custom_style(),
                       direction="vertical",
                       key="test_sim_rank")
    st.markdown("Quando estiver pronto(a) para começar, é só ir para a página Avaliação no menu lateral (à esquerda). Boa avaliação! :smile:")
elif st.session_state.authentication_status == False:
    st.error("Usuário/senha incorreta")
