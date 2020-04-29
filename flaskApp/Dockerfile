FROM python:slim

COPY . /app 
WORKDIR /app 
RUN pip install -r requirementFlask.txt
ENTRYPOINT ["python"] 
CMD ["flaskApp.py"]