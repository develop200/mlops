FROM python:3.11

# set a directory for the app
WORKDIR /usr/src/app

# copy all the files to the container
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENV SECRET_KEY=${SECRET_KEY}
ENV SQLALCHEMY_DATABASE_URI=${SQLALCHEMY_DATABASE_URI}
ENV DEBUG=${DEBUG}

# tell the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python", "./manage.py"]
