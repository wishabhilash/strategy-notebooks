FROM ubuntu:22.04

# Prevent interactive prompts during package installs and update OS
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common

# Install build tools and Python 3.10 + pip
RUN apt-get install -y python3.10 python3.10-venv python3.10-distutils python3-pip gcc make zip wget tar gzip

# Make python3 point to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
RUN tar -xzf ta-lib-0.4.0-src.tar.gz
RUN wget -O config.guess https://git.savannah.gnu.org/cgit/config.git/plain/config.guess && \
    wget -O config.sub https://git.savannah.gnu.org/cgit/config.git/plain/config.sub && \
    mv config.guess ta-lib/ && \
    mv config.sub ta-lib/
RUN cd ta-lib && ./configure --prefix=/usr --build=aarch64-linux-gnu --host=aarch64-linux-gnu && \
    make && \
    make install
COPY requirements.layer.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.layer.txt -t /lambda/python/lib/python3.10/site-packages/
RUN cp /usr/lib/libta_lib.so /lambda/python/lib/
WORKDIR /lambda
RUN rm -rf python/bin python/include python/pyvenv.cfg \
        python/**/__pycache__ \
        python/lib/python3.10/site-packages/numpy* \
        python/lib/python3.10/site-packages/pandas \
        python/lib/python3.10/site-packages/pandas-*.dist-info/ \
        python/lib/python3.10/site-packages/asyncio* \
        python/lib/python3.10/site-packages/pip* \
        python/lib/python3.10/site-packages/packaging*
RUN zip -r9 python.zip python
RUN rm -rf /lambda/python


