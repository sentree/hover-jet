#==================================================================#
# Base: Define a few important configurations for all later images #
#==================================================================#

FROM ubuntu:bionic AS base

# Set up a noninteractive environment for APT
ENV DEBIAN_FRONTEND=noninteractive

# Update the distribution packages in the image and install critical utilities
RUN apt-get update \
  && apt-get dist-upgrade -y \
  && apt-get autoremove -y --purge \
  && apt-get install --no-install-recommends -y \
    ca-certificates \
    locales \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Generate the en_US.UTF-8 locale and make it the default
RUN locale-gen en_US.UTF-8 \
  && update-locale \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Force the locale environment variables to en_US.UTF-8
# This is what Ubuntu Docker upstream recommends for UTF-8 support
ENV LANG=en_US.UTF-8 \
  LANGUAGE=en_US:en \
  LC_ALL=en_US.UTF-8


#================================================================#
# Jet: A temporary catch-all stage for the rest of the jet image #
#================================================================#

FROM base AS jet

# Since this stage persists APT list files, update them at the start
RUN apt-get update

#
# Install devtools
#

RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt-get install build-essential -y
RUN apt-get update

RUN apt-get install git -y
RUN apt-get install cmake gcc-7 g++-7 -y

RUN apt-get install libgtest-dev -y
RUN cd /usr/src/gtest; cmake CMakeLists.txt; make; cp *.a /usr/lib
RUN apt-get install i2c-tools -y
RUN apt-get install vim nano -y
RUN apt-get install gdb -y

#
# Libs
#

RUN apt-get install libxmu-dev libxi-dev -y
RUN apt-get install libassimp-dev -y
RUN apt-get install libyaml-cpp-dev -y
RUN apt-get install tzdata -y
RUN apt-get install libssl-dev -y
RUN apt-get install libreadline6-dev -y
RUN apt-get install libserial-dev -y

#
# Process management
#
RUN apt-get install supervisor -y
RUN mkdir -p /var/log/supervisor
RUN mkdir /var/log/supervisord/


#
# Python
#

RUN apt-get install pkg-config python-pip ipython -y
RUN pip install --upgrade pip
RUN apt-get install libfreetype6-dev -y
RUN pip install matplotlib
RUN pip install colorama
RUN pip install generate-cmake
RUN apt-get install python-tk -y

#
# Gl Viewer Support
#
RUN apt-get install xorg-dev libglu1-mesa-dev -y
RUN apt-get install libglfw3 libglfw3-dev libglew-dev -y
RUN apt-get install freeglut3-dev -y


#
# xvfb for x-fowarding while running headless (on a mac, server, or vehicle node)
#

RUN apt-get install -y xvfb
COPY ./image_files/etc/init.d/xvfb.init /etc/init.d/xvfb
RUN chmod +x /etc/init.d/xvfb
RUN update-rc.d xvfb defaults
CMD (service xvfb start; export DISPLAY=:10;)


#
# Mosquitto
#
RUN apt-get install mosquitto mosquitto-clients -y


#
# paho.mqtt.cpp
#

RUN apt-get install doxygen graphviz -y
RUN git clone https://github.com/eclipse/paho.mqtt.c.git; cd paho.mqtt.c; git checkout v1.2.1; cmake -Bbuild -H. -DPAHO_WITH_SSL=ON; cmake --build build/ --target install; ldconfig
RUN rm -r paho.mqtt.c
RUN git clone https://github.com/eclipse/paho.mqtt.cpp; cd paho.mqtt.cpp; cmake -Bbuild -H. -DPAHO_BUILD_DOCUMENTATION=TRUE -DPAHO_BUILD_SAMPLES=TRUE; cmake --build build/ --target install
RUN rm -r paho.mqtt.cpp


#
# GUID library
#

RUN apt-get install uuid-dev -y
RUN git clone https://github.com/graeme-hill/crossguid.git; cd crossguid; git reset --hard 0f2753174c914bc561504b14c963d179283fe829; mkdir build; cd build; cmake ..; make install
RUN rm -r crossguid


#
# X tests (xeyes, glxgears, etc.)
#

RUN apt-get install mesa-utils x11-apps -y


#
# Webcam tests (ex. fswebcam --save test.jpeg; eog test.jpeg)
#

RUN apt-get install fswebcam eog -y


#
# Adding serial library
#
RUN apt-get install -y autoconf libtool
RUN git clone https://github.com/martinling/libserialport.git; cd libserialport; ./autogen.sh; ./configure; make; make install;
RUN rm -r libserialport
RUN ldconfig


#
# Opencv
#

RUN cd /; apt-get install libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev -y
RUN git clone https://github.com/opencv/opencv_contrib.git --depth=1
RUN cd opencv_contrib; CONTRIB_PATH=$PWD; git reset --hard a17185c6dc7aa554591ad1be38923232472f8001; cd ..; git clone https://github.com/opencv/opencv.git --depth=1; cd opencv; git reset --hard e2dbf054ac5c54bd328f648d9d6146c09dfd5484; mkdir build; cd build; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=$CONTRIB_PATH/modules ..; make -j3; make install
RUN rm -rf opencv
RUN rm -rf opencv_contrib

RUN ldconfig -p
