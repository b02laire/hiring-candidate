FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install all of the needed packages
RUN pip install --no-cache-dir -r requirements.txt

## Run all detected pytest tests
#CMD ["pytest"]
