# Dockerfile-nginx
FROM nginx:latest
# Nginx will listen on this port
# Remove the default config file that
# /etc/nginx/nginx.conf includes
RUN rm /etc/nginx/conf.d/*

# We copy the requirements file in order to install
# Python dependencies
COPY app.conf /etc/nginx/conf.d