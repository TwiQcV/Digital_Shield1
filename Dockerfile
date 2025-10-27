FROM python:3.10.6-buster
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY Digital_Shield_Deployment /Digital_Shield_Deployment
COPY Digital_Shield_Packages /Digital_Shield_Package
CMD uvicorn  Digital_Shield_Deployment.fast:app --host 0.0.0.0 --port $PORT
