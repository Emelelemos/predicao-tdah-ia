
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Sistema Inteligente de Predição de TDAH",
    layout="wide"
)


# ESTILO VISUAL
st.markdown("""
<style>
.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    letter-spacing: -0.5px;
}
div[data-testid="stForm"] {
    border: 1px solid #dce3eb;
    border-radius: 16px;
    padding: 24px;
}
div[data-testid="stMetric"] {
    background-color: #eef3f9;
    padding: 20px;
    border-radius: 12px;
}
.stButton button, .stFormSubmitButton button {
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# CARREGAMENTO DO MODELO
@st.cache_resource
def carregar_modelo():
    caminho = Path(__file__).parent / "artefatos.joblib"
    return joblib.load(caminho)


artefatos = carregar_modelo()
colunas = list(artefatos["colunas_treino"])
modelo = artefatos["modelo_rf"]
imputer = artefatos["imputer"]


# APRESENTAÇÃO
st.title(" Sistema Inteligente de Predição de TDAH")

st.write(
    "Aplicação acadêmica desenvolvida com técnicas de "
    "aprendizado de máquina, utilizando o algoritmo "
    "**Random Forest** e dados do **ADHD-200**."
)

st.divider()

st.subheader("Avaliação experimental")

st.write(
    "Preencha as informações abaixo para que o modelo "
    "realize uma classificação com base nos dados informados."
)


# OPÇÕES CATEGÓRICAS
def obter_opcoes(prefixo):
    valores = [
        coluna[len(prefixo):]
        for coluna in colunas
        if coluna.startswith(prefixo)
    ]
    return sorted(valores)


opcoes_lateralidade = obter_opcoes("Handedness_")
opcoes_medicacao = obter_opcoes("Med Status_")


# FORMULÁRIO
with st.form("formulario_tdah"):

    st.subheader("1. Informações do participante")

    col1, col2, col3 = st.columns(3)

    with col1:
        idade = st.number_input(
            "Idade (anos)",
            min_value=1,
            max_value=100,
            value=12,
            step=1,
            format="%d"
        )

    with col2:
        genero = st.number_input(
            "Gênero — código",
            min_value=0,
            max_value=1,
            value=1,
            step=1,
            format="%d",
            help="Codificação numérica utilizada no treinamento."
        )

    with col3:
        centro = st.number_input(
            "Centro de coleta — código",
            min_value=0,
            value=5,
            step=1,
            format="%d",
            help="Identificador do centro de coleta dos dados."
        )

    st.divider()

    st.subheader("2. Avaliação cognitiva")

    st.caption(
        "Informe as pontuações das avaliações cognitivas disponíveis."
    )

    instrumento = st.number_input(
        "Instrumento de QI — código",
        min_value=0,
        value=2,
        step=1,
        format="%d",
        help="Identificador do instrumento utilizado na avaliação."
    )

    col4, col5 = st.columns(2)

    with col4:
        qi_verbal = st.number_input(
            "QI verbal",
            min_value=0,
            max_value=200,
            value=100,
            step=1,
            format="%d"
        )

        qi_total2 = st.number_input(
            "QI total — Full2 IQ",
            min_value=0,
            max_value=200,
            value=100,
            step=1,
            format="%d"
        )

    with col5:
        qi_desempenho = st.number_input(
            "QI de desempenho",
            min_value=0,
            max_value=200,
            value=100,
            step=1,
            format="%d"
        )

        qi_total4 = st.number_input(
            "QI total — Full4 IQ",
            min_value=0,
            max_value=200,
            value=100,
            step=1,
            format="%d"
        )

    st.divider()

    st.subheader("3. Informações complementares")

    col6, col7 = st.columns(2)

    with col6:
        lateralidade = st.selectbox(
            "Lateralidade",
            options=opcoes_lateralidade,
            help="Valor registrado para preferência manual."
        )

    with col7:
        medicacao = st.selectbox(
            "Situação de medicação",
            options=opcoes_medicacao,
            help="Valor registrado para situação de medicação."
        )

    st.write("")

    enviar = st.form_submit_button(
        "Realizar predição",
        type="primary",
        use_container_width=True
    )


# PREDIÇÃO
if enviar:

    # Criar registro com a estrutura utilizada no treinamento
    registro = {
        coluna: 0.0
        for coluna in colunas
    }

    registro.update({
        "Site": float(centro),
        "Gender": float(genero),
        "Age": float(idade),
        "IQ Measure": float(instrumento),
        "Verbal IQ": float(qi_verbal),
        "Performance IQ": float(qi_desempenho),
        "Full2 IQ": float(qi_total2),
        "Full4 IQ": float(qi_total4)
    })

    # Aplicar codificação categórica
    for prefixo, valor in [
        ("Handedness_", lateralidade),
        ("Med Status_", medicacao)
    ]:
        chave = prefixo + str(valor).strip().upper()

        if chave in registro:
            registro[chave] = 1.0

    # Organizar dados
    entrada = pd.DataFrame(
        [registro],
        columns=colunas
    )

    # Aplicar imputação do treinamento
    dados_processados = imputer.transform(entrada)

    # Realizar classificação
    previsao = int(
        modelo.predict(dados_processados)[0]
    )

    st.divider()
    st.subheader("Resultado da análise")

    if previsao == 1:
        resultado = "Classe 1 — TDAH"
    else:
        resultado = "Classe 0 — Sem TDAH"

    st.metric(
        label="Classificação prevista",
        value=resultado
    )

    if hasattr(modelo, "predict_proba"):
        classes = list(modelo.classes_)

        if 1 in classes:
            indice = classes.index(1)

            pontuacao = float(
                modelo.predict_proba(
                    dados_processados
                )[0][indice]
            )

            st.write(
                "**Pontuação estimada para a classe 1:** "
                f"{pontuacao:.1%}"
            )

            st.progress(pontuacao)

    st.success(
        "Análise experimental concluída."
    )


# RODAPÉ
st.divider()

st.caption(
    "Protótipo acadêmico desenvolvido com dados do ADHD-200. "
    "As classificações são experimentais e não constituem "
    "diagnóstico clínico de TDAH."
)
