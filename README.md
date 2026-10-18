# MVP de predição experimental de TDAH

Aplicação acadêmica Streamlit que usa o Random Forest salvo em `artefatos.joblib`.

## Executar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para deploy no Streamlit Community Cloud, envie `app.py`, `artefatos.joblib` e `requirements.txt` para a raiz do repositório GitHub. Escolha `app.py` como arquivo principal. Recomenda-se Python 3.12.

**Limitações:** Os códigos categóricos devem ser conferidos no dicionário do ADHD-200. O modelo não foi validado para diagnóstico ou triagem clínica. Não insira dados identificáveis de pacientes.
