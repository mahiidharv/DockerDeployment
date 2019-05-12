FROM python:3.7.2-stretch

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app 
ADD . /app

# Install the dependencies
RUN pip install -r requirements.txt

# run the command to start uWSGI
#CMD ["uwsgi", "app.ini"]
#CMD ["gunicorn"  , "-b", "0.0.0.0:5000", "wsgi:app"]