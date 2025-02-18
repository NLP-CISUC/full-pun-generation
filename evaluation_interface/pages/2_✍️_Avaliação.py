import polars as pl
import streamlit as st
from utils import get_results_path, load_config, load_data
from streamlit.components.v1 import html

if "username" not in st.session_state or st.session_state.username is None:
    st.error("Por favor, faça login primeiro.")
    st.stop()


username = st.session_state.username
cfg = load_config()
splits = cfg["splits"]

results_path = get_results_path(username)
results_path.parent.mkdir(parents=True, exist_ok=True)

df = load_data().filter(pl.col("headline_id").is_in(splits[username]))


def format_headline():
    cur_headline = st.session_state.cur_headline
    return df.filter(pl.col("headline_id") == cur_headline)["headline"].first()


def update_jokes():
    cur_headline = st.session_state.cur_headline
    if ("results_df" in st.session_state and
        cur_headline in st.session_state.results_df["headline_id"]):
        jokes = (st.session_state.results_df
                 .filter(pl.col("headline_id") == cur_headline)["generated"]
                 .to_list())
    else:
        jokes = (df.filter(pl.col("headline_id") == cur_headline)["generated"]
                 .unique()
                 .sample(fraction=1, shuffle=True)
                 .to_list())
    st.session_state.cur_jokes = jokes


def update_current_headline():
    st.session_state.cur_headline = splits[username][st.session_state.cur_idx]


def decrease_index():
    st.session_state.cur_idx -= 1
    update_current_headline()
    update_jokes()
    st.session_state.scroll_to_top = True


def increase_index():
    st.session_state.cur_idx += 1
    update_current_headline()
    update_jokes()
    st.session_state.scroll_to_top = True


def set_show_success():
    st.session_state.show_success = True


def update_rates():
    jokes = {}
    for i in range(len(st.session_state.cur_jokes)):
        fun = st.session_state[f"funniness_{st.session_state.cur_headline}_{i}"]
        rel = st.session_state[f"relation_{st.session_state.cur_headline}_{i}"]
        jokes[st.session_state.cur_jokes[i]] = {"funniness": fun, "relation": rel}

    st.session_state.rates[st.session_state.cur_headline] = jokes


#
# ------ Initialize state ------
#
if "rates" not in st.session_state:
    st.session_state.rates = {}
if "results_df" not in st.session_state and results_path.exists():
    st.session_state.results_df = pl.read_ndjson(results_path)
    for row in st.session_state.results_df.iter_rows(named=True):
        if row["headline_id"] not in st.session_state.rates:
            st.session_state.rates[row["headline_id"]] = {}
        st.session_state.rates[row["headline_id"]][row["generated"]] = {
            "funniness": row["funniness"],
            "relation": row["relation"]
        }
if "cur_idx" not in st.session_state:
    st.session_state.cur_idx = 0
if "cur_headline" not in st.session_state:
    st.session_state.cur_headline = splits[username][st.session_state.cur_idx]
if "show_success" not in st.session_state:
    st.session_state.show_success = False
if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False
if "cur_jokes" not in st.session_state:
    update_jokes()


#
# ------ Streamlit page ------
#
if st.session_state.scroll_to_top:
    html(f"""
    <script>
        console.log("Scrolling to top");
        var element = window.parent.document.querySelector(".stMainBlockContainer");
        element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    </script>
    <!-- Unique marker: {st.session_state.cur_idx} -->
    """, height=0)
    st.session_state.scroll_to_top = False


if st.session_state.show_success:
    st.success("Avaliação concluída! Muito obrigado por participar! :smile:")
    st.stop()

header = st.container()
header.header(format_headline(), anchor="headline")
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
st.markdown(
    """
<style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 2.875rem;
        background-color: #F5F5F5;
        z-index: 999;
    }
    .fixed-header {
        border-bottom: 1px solid #333333;
    }
    .stMainBlockContainer {
        max-width: 850px;
    }
    li::marker {
        font-size: 19px;
    }
</style>
    """,
    unsafe_allow_html=True
)

for i in range(len(st.session_state.cur_jokes)):
    try:
        cur_headline = st.session_state.cur_headline
        cur_joke = st.session_state.cur_jokes[i]
        default_fun = st.session_state.rates[cur_headline][cur_joke]["funniness"]
        default_rel = st.session_state.rates[cur_headline][cur_joke]["relation"]
    except (KeyError, IndexError):
        default_fun = "Não tem piada"
        default_rel = "Não tem relação"

    with st.container(border=True):
        st.markdown(f"<p style=\"font-size:19px\">{st.session_state.cur_jokes[i]}</p>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.pills("1\. Qual o nível de piada?",
                     ["Não tem piada", "Tem pouca piada",
                      "Tem piada"],
                     default=default_fun,
                     key=f"funniness_{st.session_state.cur_headline}_{i}",
                     on_change=update_rates)
        with col2:
            st.pills("2\. A piada tem relação com a notícia?",
                     ["Não tem relação", "Tem pouca relação",
                      "Tem relação"],
                     default=default_rel,
                     key=f"relation_{st.session_state.cur_headline}_{i}",
                     on_change=update_rates)

    st.container(height=25, border=False)

btn_col1, _, btn_col2 = st.columns([1, 3, 1])
with btn_col1:
    first_example = (st.session_state.cur_idx == 0)
    st.button("Anterior", disabled=first_example,
              on_click=decrease_index,
              use_container_width=True)
with btn_col2:
    last_example = (st.session_state.cur_idx == len(splits[username]) - 1)
    if not last_example:
        st.button("Próximo", on_click=increase_index, use_container_width=True)
    else:
        st.button("Concluir", type="primary", on_click=set_show_success,
                  use_container_width=True)

#
# ------ Consolidate and save results ------
#
if st.session_state.rates and st.session_state.cur_headline in st.session_state.rates:
    rates = {"evaluator": [], "headline_id": [], "generated": [], "funniness": [], "relation": []}
    for headline_id in st.session_state.rates:
        for generated in st.session_state.rates[headline_id]:
            rates["evaluator"].append(username)
            rates["headline_id"].append(headline_id)
            rates["generated"].append(generated)
            rates["funniness"].append(st.session_state.rates[headline_id][generated]["funniness"])
            rates["relation"].append(st.session_state.rates[headline_id][generated]["relation"])
    rates_df = pl.DataFrame(rates)
    rates_df = df.join(rates_df, on=["headline_id", "generated"])
    if "results_df" in st.session_state:
        rates_df = (st.session_state.results_df.update(rates_df,
                                                       on=["headline_id",
                                                           "generated"],
                                                       how="full"))
    st.session_state.results_df = rates_df
    st.session_state.results_df.write_ndjson(results_path)
