FROM python:slim

COPY . /app 
WORKDIR /app 
RUN pip install -r requirement.txt
ENTRYPOINT ["python"] 
CMD ["manage.py runserver"]