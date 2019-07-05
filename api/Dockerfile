FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip install --proxy=http://proxy.ch.abb.com:8080 -r requirements.txt
ENTRYPOINT ["python"]
CMD ["api.py"]