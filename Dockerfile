FROM ultralytics/ultralytics:latest

WORKDIR /code

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip

COPY ./OBB /code/OBB

RUN pip install --upgrade pip
RUN pip install -r /code/OBB/requirements.txt
RUN pip install fastapi

RUN export PYTHONPATH=$PYTHONPATH:/code:/code

EXPOSE 8081

CMD ["fastapi", "run", "OBB/app.py", "--port", "8081"]
