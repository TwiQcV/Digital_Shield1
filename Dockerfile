FROM python:3.10.6-buster
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY Digital_Shield_Deployment /Digital_Shield_Deployment
CMD uvicorn  Digital_Shield_Deployment.app:app --host 0.0.0.0 --port 8080
