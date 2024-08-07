FROM ultralytics/ultralytics@sha256:81923166890bcd9f5545033ca811eba31457b5ae0f46bd4c0e9d874722a590e2

WORKDIR /code

COPY ./OBB /code/OBB

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv


#python3-pip

#RUN pip install --upgrade pip

RUN conda env create -f /code/OBB/environment.yml -v

SHELL ["conda", "run", "-n", "itdp", "/bin/bash", "-c"]

#RUN conda activate itdp && pip install -r /code/OBB/requirements.txt

RUN export PYTHONPATH=$PYTHONPATH:/code

EXPOSE 8081

#CMD ["conda", "run", "--no-capture-output", "-n", "itdp", "fastapi", "run", "OBB/app.py", "--port", "8081"]
CMD ["sleep", "60"]
