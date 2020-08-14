FROM python:3.6-slim

EXPOSE 8501

WORKDIR /app

# This is needed to run open-cv in python slim package.
RUN apt-get update && apt-get -y install libgl1-mesa-glx; apt-get clean

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py"]