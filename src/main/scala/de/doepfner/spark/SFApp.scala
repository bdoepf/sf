package de.doepfner.spark

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType

object SFApp {
  def main(args: Array[String]): Unit = {

    implicit val spark: SparkSession = SparkSession.builder().master("local[*]").getOrCreate()
    import spark.implicits._
    val inputDf = spark
      .read
      .option("header", "true")
      .csv("data/train.csv")
      .drop("Dates", "Descript", "Resolution")
      .drop("Address")
      .withColumnRenamed("Category", "label")
      .withColumn("X", $"X".cast(DoubleType))
      .withColumn("Y", $"Y".cast(DoubleType))

    inputDf.printSchema()

    inputDf.cache()

    // Label, crime categories
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(inputDf)

    val dayOfWeekIndexer = new StringIndexer()
      .setInputCol("DayOfWeek")
      .setOutputCol("indexedDayOfWeek")
      .fit(inputDf)

    val pdDistrictIndexer = new StringIndexer()
      .setInputCol("PdDistrict")
      .setOutputCol("indexedPdDistrict")
      .fit(inputDf)

    val assembler = new VectorAssembler()
      .setInputCols(Array("indexedDayOfWeek", "indexedDayOfWeek", "indexedPdDistrict", "X", "Y"))
      .setOutputCol("features")

    // Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(100) // features with > distinct values are treated as continuous.
     // .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = inputDf.randomSplit(Array(0.7, 0.3))

    // Train a DecisionTree model.
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, dayOfWeekIndexer, pdDistrictIndexer, assembler, featureIndexer, dt, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(100)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    val treeModel = model.stages(5).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
  }
}
