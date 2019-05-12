

FROM mysql:5.7

COPY ./predictions.sql /tmp/

ENV MYSQL_DATABASE insofe_customerdata

COPY ./predictions.sql  /docker-entrypoint-initdb.d/