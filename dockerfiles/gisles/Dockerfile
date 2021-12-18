FROM gfcn:base

ARG CODE_DIR="gfcn"

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
# COPY $CODE_DIR /workspace/gfcn
# RUN chown -R 42420:42420 /workspace
#WORKDIR /workspace/gfcn

COPY run.sh /workspace/gfcn/run.sh
RUN mkdir -p /workspace/gfcn/$output
RUN chmod +x /workspace/gfcn/run.sh 

ENTRYPOINT []
CMD ["/bin/bash", "run.sh"]
