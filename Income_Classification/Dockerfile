FROM python:3.9

WORKDIR /app

#RUN apt-get update && apt-get install -y ffmpeg

COPY app/streamlit /app/streamlit

COPY requirements.txt /app/requirements.txt
# COPY ./requirements.txt Income_Classification/app/requirements.txt


RUN pip install --no-cache-dir -r /app/requirements.txt

# EXPOSE 8501
EXPOSE 8080

CMD ["streamlit", "run", "streamlit/income.py", "--server.port", "8080", "--server.address", "0.0.0.0"]

# scp app_build_001.tar remote_username@172.10.10.2:/remote/directory
# scp -i /Users/patil/Workspace/VS\ code/Income_Classification/income-server-key.pem app_build_001.tar ec2-user@ec2-18-222-116-60.us-east-2.compute.amazonaws.com:/home/ec2-user/