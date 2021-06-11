FROM python:3.7

WORKDIR /horse-racing-tools
ENV LANG C.UTF-8

RUN pip install --upgrade pip --no-cache-dir

COPY requirements.txt /horse-racing-tools/
RUN pip install -r requirements.txt

COPY . /horse-racing-tools/

ENV PYTHONPATH "${PYTHONPATH}:/horse-racing-tools/"
