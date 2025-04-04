FROM python:3.11-slim AS base

RUN apt-get update -y && apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY . ./

RUN pip install -r requirements.txt

#ENV DOCKERIZE_VERSION v0.9.3
#
#RUN apt-get update \
#    && apt-get install -y wget \
#    && wget -O - https://github.com/jwilder/dockerize/releases/download/$DOCKERIZE_VERSION/dockerize-linux-amd64-$DOCKERIZE_VERSION.tar.gz | tar xzf - -C /usr/local/bin \
#    && apt-get autoremove -yqq --purge wget && rm -rf /var/lib/apt/lists/*


#CMD dockerize -wait=http://ollama:11434 -timeout 60m exec uvicorn app:app --host '0.0.0.0' --port 5005
# CMD exec uvicorn app:app --host '0.0.0.0' --port 5005 --reload