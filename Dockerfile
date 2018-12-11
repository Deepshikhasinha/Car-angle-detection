# Dockerfile - this is a comment. Delete me if you want.
FROM ubuntu:16.04
RUN apt-get update
RUN apt install unzip
RUN apt-get install python-pip python-dev python-virtualenv -y
RUN apt-get install apache2 -y
RUN apt-get install libapache2-mod-wsgi -y
RUN pip install flask
RUN pip install -U flask-cors
RUN pip install pillow
RUN pip install flask-bootstrap
RUN pip install boto3
RUN pip install -U tensorflow
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
COPY ./000-default.conf /etc/apache2/sites-enabled/000-default.conf
RUN service apache2 restart
ENTRYPOINT ["python"]
CMD ["app.py"]
EXPOSE 5000
