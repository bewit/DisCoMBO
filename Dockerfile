ARG baseimage
FROM ${baseimage} 
# GPU: any CUDA base image that ships Python 3.10 (spflow>=1.0 needs Python>=3.10; jahs/HPOBench need <3.11)
# python:3.10-slim -- for CPU
ENV DEBIAN_FRONTEND noninteractive
WORKDIR /app/

COPY ./requirements.txt .
COPY ./setup.sh /app/setup.sh
COPY ./NASLib/ /app/NASLib/
COPY ./HPOBench/ /app/HPOBench/
COPY ./jahs_bench_201/ /app/jahs_bench_201/
COPY ./ConfigSpace/ /app/ConfigSpace/

RUN echo

RUN apt-get update && apt-get install -y tmux && apt-get install -y git && apt-get install -y build-essential && apt-get install -y liblapack-dev && apt-get install -y libblas-dev && apt-get install -y gfortran && apt-get install -y tk && apt-get install -y numactl && apt-get install -y coreutils && apt-get install -y wget && apt-get install unzip && apt-get install -y python3-pkg-resources
RUN pip install build

# RUN cd ./NASLib && pip install --upgrade pip setuptools wheel
RUN cd ./NASLib && pip install --upgrade setuptools wheel
# RUN cd ./NASLib && pip install -e .
RUN cd ./NASLib && pip install .
RUN cd ./NASLib && python setup.py bdist_wheel
# RUN cd ./NASLib && python -m build
RUN pip install ./NASLib/dist/naslib-0.1.0-py3-none-any.whl
RUN cd ./HPOBench && pip install .
RUN cd ./jahs_bench_201 && pip install -e .

RUN pip install -r requirements.txt

# remove ConfigSpace if installed
RUN pip uninstall -y ConfigSpace
RUN cd ./ConfigSpace && python -m build && pip install dist/configspace-0.7.1-*.whl 
RUN chmod +x /app/setup.sh

CMD ["/app/setup.sh"]
