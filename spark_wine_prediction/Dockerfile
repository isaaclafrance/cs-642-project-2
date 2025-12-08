FROM eclipse-temurin:17-jre

RUN apt-get update && apt-get install -y curl bash procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Install Apache Spark

ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3

RUN curl -L "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
      -o /tmp/spark.tgz && \
    tar -xzf /tmp/spark.tgz -C /opt/ && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm /tmp/spark.tgz

ENV SPARK_HOME=/opt/spark
ENV PATH="${PATH}:${SPARK_HOME}/bin:${SPARK_HOME}/sbin"


WORKDIR /spark_prediction

# Copy trained model directory
COPY wine_quality_model /spark_prediction/wine_quality_model

# Copy prediction JAR
COPY target/sparkwinequalitypredictor-1.0.0.jar /spark_prediction/predictor.jar

ENTRYPOINT ["spark-submit", "--master", "local[*]", "--class", "org.cloud_computing.SparkWineQualityPredictor", "/spark_prediction/predictor.jar", "/spark_prediction/wine_quality_model"]