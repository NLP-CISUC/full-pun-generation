import streamlit as st
import streamlit_authenticator as stauth


authenticator = stauth.Authenticate("config/credentials.yaml")
try:
    authenticator.login()
except Exception as e:
    st.error(e)

st.markdown("""
<style>
    .stMainBlockContainer {
        max-width: 850px;
    }
</style>
""", unsafe_allow_html=True)

if st.session_state.authentication_status:
    authenticator.logout("Terminar sessão", "main")
    st.title("Avaliação de trocadilhos")
    st.markdown(f"""
    Bem-vindo(a) *{st.session_state.name}*!

    Obrigado por ter aceitado em participar desta avaliação. O objetivo é analisar a qualidade de piadas geradas automaticamente a partir de títulos de notícias.

    Essa avaliação está estruturada da seguinte forma. Você receberá um máximo de 5 notícias, uma por vez, em uma página com o título no topo e 10 piadas listadas abaixo. Cada piada possui duas escalas, correspondentes aos critérios abaixo. Para realizar a avaliação, você deverá selecionar o valor correspondente à sua percepção de acordo com o critério.

    As escalas seguem dois critérios diferentes:
    - **Nível de humor** (escala da **esquerda**)
        - "Não tem piada" = Não imagino ninguém rindo disso
        - "Tem pouca piada" = O texto pode fazer sorrir
        - "Tem piada" = O texto pode fazer rir
    - **Relação com o título** (escala da **direita**)
        - "Não relacionado à notícia" = A piada não tem relação alguma com o título ou não faz sentido no contexto
        - "Pouco relacionado à notícia" = A piada tem pouca relação com o título (por exemplo, alguma palavra-chave)
        - "Relacionado à notícia" = A piada tem relação com o tema geral do título

    Aqui embaixo você verá um pequeno exemplo (com 3 piadas) para entender como é a interface de avaliação, pode brincar à vontade! Se tiver qualquer dúvida, pode vir falar comigo.
    """)

    joke_examples = ["Piada 1", "Piada 2", "Piada 3"]
    st.header("ChatGPT é visto dirigindo Uber após chegada de IA chinesa")
    for i in range(len(joke_examples)):
        with st.container(border=True):
            st.markdown(f"<p style=\"font-size:19px\">{joke_examples[i]}</p>",
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.pills("1\. Qual o nível de piada?",
                         ["Não tem piada", "Tem pouca piada", "Tem piada"],
                         default="Não tem piada",
                         key=f"test_slider_fun_{i}")
            with col2:
                st.pills("2\. A piada tem relação com a notícia?",
                         ["Não tem relação",
                          "Tem pouca relação",
                          "Tem relação"],
                         default="Não tem relação",
                         key=f"test_slider_rel_{i}")

    st.markdown("Quando estiver pronto(a) para começar, é só ir para a página Avaliação no menu lateral (à esquerda). Boa avaliação! :smile:")
elif st.session_state.authentication_status == False:
    st.error("Usuário/senha incorreta")
