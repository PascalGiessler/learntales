FROM python:3.12-slim-bookworm

WORKDIR /svc

COPY pyproject.toml /svc

RUN pip install poetry

RUN poetry config virtualenvs.create false \
&& poetry install --no-dev --no-interaction --no-ansi

COPY ./ /svc/app/

EXPOSE 8501

ENTRYPOINT ["python", "-m", "streamlit", "run", "./app/lit/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
