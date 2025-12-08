package org.cloud_computing;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import java.nio.file.Files;
import java.nio.file.Path;
import java.io.IOException;
import java.util.Arrays;
import java.util.List; 

public class SparkWineQualityPredictor {

    public static SparkSession getSparkSession() {
        return SparkSession.builder()
                .appName("WineQualityPredictor")
                .master("local[*]")
                .getOrCreate();
    }

    public static Dataset<Row> loadData(SparkSession sparkSession, String fileName) {
        Dataset<Row> fileData = sparkSession.read()
                .option("inferSchema", "true")
                .option("header", "true")
                .option("sep", ";")
                .csv(fileName);

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
                .withColumnRenamed("quality", "label")
                .select("label", "wine_quality_features");
    }

    public static Dataset<Row> loadVectorizedTestData(SparkSession sparkSession, String fileLocation) {
        Dataset<Row> raw = loadData(sparkSession, fileLocation);
        return vectorizedData(raw);
    }

    public static double calculateF1Score(LogisticRegressionModel model, Dataset<Row> data) {

        Dataset<Row> modelPredictions = model.transform(data);

        MulticlassClassificationEvaluator evaluator =
                new MulticlassClassificationEvaluator()
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setMetricName("f1");

        return evaluator.evaluate(modelPredictions);
    }

    public static String retrieveAnyMountedDirectory() throws IOException {
        //finds users's mounted directory if aany had been specified while running in docker container

        List<String> lines = Files.readAllLines(Path.of("/proc/mounts"));

        for (String line : lines) {
            String[] parts = line.split(" ");
            if (parts.length >= 3) {
                String mountPoint = parts[1];

                if (!mountPoint.startsWith("/proc")
                        && !mountPoint.startsWith("/sys")
                        && !mountPoint.startsWith("/dev")
                        && !mountPoint.startsWith("/etc")
                        && !mountPoint.equals("/")) {
                    return mountPoint + "/";
                }
            }
        }

        return "./"; //default to current working directory
    }

    public static void saveF1Score(double score) {
        //Save f1 score to a text file
        try {
            String scoreString = String.valueOf(score);

            String saveDirectory = retrieveAnyMountedDirectory();
            Path scoreFilePath = Path.of(saveDirectory + "f1Score.txt");

            Files.writeString(scoreFilePath, scoreString);

        } catch (Exception ex) {
            System.out.println("Error while saving F1 Score: " + ex);
        }

        System.out.println(String.format(
                "\u001B[1m\u001B[32m\n\n\n *** F1 SCORE: %s *** \n\n\n\u001B[0m",
                score));
    }

    public static void main(String[] args) throws IOException {

        String testDataFileLocation;
        String trainedModelFileLocation;

        if (args.length >= 2) {
            testDataFileLocation = args[1];
        } else {
            testDataFileLocation = "/home/ubuntu/spark_prediction/ValidationDataset.csv";
        }

        if (args.length >= 1) {
            trainedModelFileLocation = args[0];
        } else {
            trainedModelFileLocation = "/home/ubuntu/spark_prediction/wine_quality_model";
        }

        SparkSession sparkSession = getSparkSession();

        Dataset<Row> testData = loadVectorizedTestData(sparkSession, testDataFileLocation);

        LogisticRegressionModel trainedModel = LogisticRegressionModel.load(trainedModelFileLocation);

        double f1Score = calculateF1Score(trainedModel, testData);

        saveF1Score(f1Score);

        sparkSession.stop();
    }
}