package org.cloud_computing;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;

import org.apache.spark.ml.feature.VectorAssembler;

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;

import java.io.IOException;
import java.util.Arrays;

public class SparkWineQualityTrainer {

    public static SparkSession getSparkSession() {
        return SparkSession.builder()
                .appName("WineQualityTrainerCluster")
                .getOrCreate();
    }

    public static Dataset<Row> loadData(SparkSession sparkSession, String fileName){
        Dataset<Row> fileData = sparkSession.read()
                .option("inferSchema", "true")
                .option("header", "true")
                .option("sep", ";")
                .csv(fileName);

        //Make sure to remove unnecessary quote in header names        
        String[] cleanedColumnHeaders = Arrays.stream(fileData.columns())
                .map(col -> col.replace("\"", ""))
                .toArray(String[]::new);

        return fileData.toDF(cleanedColumnHeaders);
    }

    public static Dataset<Row> vectorizedData(Dataset<Row> data) {

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{
                        "fixed acidity", "volatile acidity",
                        "citric acid", "residual sugar", "chlorides",
                        "free sulfur dioxide", "total sulfur dioxide",
                        "density", "pH", "sulphates", "alcohol"
                })
                .setOutputCol("wine_quality_features");

        return vectorAssembler.transform(data)
                .withColumnRenamed("quality", "label");
    }

    public static Dataset<Row> loadVectorizedTrainingData(SparkSession sparkSession, String fileLocation){

        Dataset<Row> trainingData = loadData(sparkSession, fileLocation);
        Dataset<Row> vectorizedTrainingData = vectorizedData(trainingData);

        return vectorizedTrainingData.select("label", "wine_quality_features");
    }

    public static Dataset<Row> loadVectorizedValidationData(SparkSession sparkSession, String fileLocation){

        Dataset<Row> validationData = loadData(sparkSession, fileLocation);
        Dataset<Row> vectorizedValidationData = vectorizedData(validationData);

        return vectorizedValidationData.select("label", "wine_quality_features");
    }

    public static LogisticRegressionModel trainModel(Dataset<Row> data) {

        LogisticRegression logisticRegression = new LogisticRegression()
                .setMaxIter(50)
                .setRegParam(0.01)
                .setFeaturesCol("wine_quality_features")
                .setLabelCol("label");

        return logisticRegression.fit(data);
    }

    public static LogisticRegressionModel optimizeModelWithValidationData(
            Dataset<Row> trainingData,
            Dataset<Row> validationData) {

        double[] optimizationParams = {0.01, 0.05, 0.1, 0.2};

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double bestEvaluationScore = -1.0;
        LogisticRegressionModel bestModel = null;
        LogisticRegressionModel model = null;
        
        //Iteratively try to find the best regresion parameter 
        for (double param : optimizationParams) {
            LogisticRegression lr = new LogisticRegression()
                    .setRegParam(param)
                    .setMaxIter(50)
                    .setLabelCol("label")
                    .setFeaturesCol("wine_quality_features");

            model = lr.fit(trainingData);

            double evaluationScore = evaluator.evaluate(model.transform(validationData));

            if (evaluationScore> bestEvaluationScore) {
                bestEvaluationScore = evaluationScore;
                bestModel = model;
            }
        }

        return bestModel != null ? bestModel : model;
    }

    public static void main(String[] args) throws IOException {
        String trainingDataFileLocation;
        String validationDataFileLocation;
        String trainedModelSaveLocation;

        if (args.length >= 1) {
            trainingDataFileLocation = args[0];
        } else {
            trainingDataFileLocation = "/home/ubuntu/spark_training/TrainingDataset.csv";
        }

        if (args.length >= 2) {
            validationDataFileLocation = args[1];
        } else {
            validationDataFileLocation = "/home/ubuntu/spark_training/ValidationDataset.csv";
        }

        if (args.length >= 3) {
            trainedModelSaveLocation = args[2];
        } else {
            trainedModelSaveLocation = "/home/ubuntu/spark_training/wine_quality_model";
        }

        SparkSession sparkSession = getSparkSession();

        Dataset<Row> scaledVectorizedTrainingData =
                loadVectorizedTrainingData(sparkSession, trainingDataFileLocation);

        Dataset<Row> scaledVectorizedValidationData =
                loadVectorizedValidationData(sparkSession, validationDataFileLocation);


        LogisticRegressionModel optimizedTrainedModel =
                optimizeModelWithValidationData(
                        scaledVectorizedTrainingData,
                        scaledVectorizedValidationData
                );

        optimizedTrainedModel.write().overwrite().save(trainedModelSaveLocation);

        sparkSession.stop();
    }
}