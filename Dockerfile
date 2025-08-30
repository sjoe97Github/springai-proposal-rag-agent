## Use the official PostgreSQL image with Alpine Linux
#FROM postgres:16-alpine
#
## Set the working directory
#WORKDIR /usr/src/pgvector
#
## Install build dependencies, including clang
##RUN apk add --no-cache git build-base postgresql-dev clang \
##    && git clone --branch v0.6.2 https://github.com/pgvector/pgvector.git \
##    && cd pgvector \
##    && make \
##    && make install \
##    && cd .. \
##    && rm -rf pgvector \
##    && apk del git build-base postgresql-dev
#
##RUN apk add --no-cache git build-base postgresql-dev clang llvm-dev \
##  && git clone --branch v0.6.2 https://github.com/pgvector/pgvector.git \
##  && make -C pgvector CLANG=clang \
##  && make -C pgvector install \
##  && rm -rf pgvector \
##  && apk del git build-base postgresql-dev clang llvm-dev
#
#RUN apk add --no-cache git build-base postgresql-dev clang llvm-dev \
#  && ln -s /usr/bin/clang /usr/bin/clang-19 \
#  && git clone --branch v0.6.2 https://github.com/pgvector/pgvector.git \
#  && make -C pgvector \
#  && make -C pgvector install \
#  && rm -rf pgvector \
#  && apk del git build-base postgresql-dev clang llvm-dev

# Build pgvector against a Postgres built with --with-llvm (Debian)
# Dockerfile
FROM postgres:16-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates postgresql-16-pgvector \
  && rm -rf /var/lib/apt/lists/*

# optional: init script that does CREATE EXTENSION on first init
COPY ./init-pgvector.sql /docker-entrypoint-initdb.d/

# Set the default command to start the database
#CMD ["postgres"]
