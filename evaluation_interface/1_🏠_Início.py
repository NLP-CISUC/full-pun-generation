import streamlit as st
import streamlit_authenticator as stauth

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

    Essa avaliação esta estuturada da seguinte forma. Você receberá um máximo de 5 noticias, uma por vez, em uma página com o título no topo e 10 piadas listadas abaixo. Cada piada possui duas escalas, correspondentes aos critérios abaixo. Para realizar a avaliação, você deverá selecionar o valor correspondente à sua percepção de acordo com o critério.

    As escalas seguem dois critérios diferentes:
    - **Nível de humor** (escala da **esquerda**, vai de "tem pouca piada" até "tem muita piada");
    - **Relação com o título** (escala da **direita**, vai de "não tem relação" até "tem muita relação").

    Aqui embaixo você verá um pequeno exemplo (com 3 piadas) para entender como é a interface de avaliação, pode brincar à vontade! Se tiver qualquer dúvida, pode vir falar comigo.
    """)

    joke_examples = ["Piada 1", "Piada 2", "Piada 3"]
    with st.expander("Avaliação de teste", expanded=True):
        st.title("ChatGPT é visto dirigindo Uber após chegada de IA chinesa")
        for i in range(len(joke_examples)):
            with st.container(border=True):
                st.markdown(f"<p style=\"font-size:21px\">{joke_examples[i]}</p>",
                            unsafe_allow_html=True)
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.select_slider("Qual o nível de piada?",
                                     ["Não tem piada", "Tem pouca piada",
                                      "Tem piada", "Tem muita piada"],
                                     key=f"test_slider_fun_{i}",
                                     label_visibility="hidden")
                with col2:
                    st.select_slider("A piada tem relação com a notícia?",
                                     ["Não tem relação", "Tem pouca relação",
                                      "Tem relação", "Tem muita relação"],
                                     key=f"test_slider_rel_{i}",
                                     label_visibility="hidden")
    st.markdown("Quando estiver pronto(a) para começar, é só ir para a página Avaliação no menu lateral (à esquerda). Boa avaliação! :smile:")
elif st.session_state.authentication_status == False:
    st.error("Usuário/senha incorreta")
