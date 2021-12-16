#FROM gfcn:v1
FROM ubuntu

ARG CODE_DIR="../../gfcn/"

ARG batch="4"
ARG lr="1E-6"
ARG tid="Pre4Ch"
ARG output="experiment3_depth"
ARG epochs="100"
ARG net="GFCNB"

ENV BATCH=$batch
ENV LR=$lr
ENV TID=$tid
ENV OUTPUT=$output
ENV EPOCHS=$epochs
ENV NET=$net

# Permision to run the OVHcloud user
#COPY $CODE_DIR /workspace/gfcn
#RUN chown -R 42420:42420 /workspace
#WORKDIR /workspace/gfcn

ENTRYPOINT []
#CMD ["/workspace/gfcn/depth_training.sh 4ch 1"]
#CMD ["python training.py -s GISLES2018 -n $NET -b $BATCH -c DCSsigmoid  -t True -N 10 --id $TID -lr $EPOCHS -g $EPOCHS --mod TMAX CBF CBV MTT --postnorm False -D $OUTPUT>> \"${OUTPUT}/gfcnb_${TID}.log"]
CMD = ["echo 'holamudo'"]
