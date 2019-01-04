package de.doepfner.spark

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType

object SFApp {
  def main(args: Array[String]): Unit = {

    implicit val spark: SparkSession = SparkSession.builder().master("local[*]").getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("WARN")

    val inputDf = spark
      .read
      .option("header", "true")
      .csv("data/train.csv")
      .drop("Dates", "Descript", "Resolution")
      .drop("Address")
      .withColumnRenamed("Category", "label")
      .withColumn("X", $"X".cast(DoubleType))
      .withColumn("Y", $"Y".cast(DoubleType))

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
    val Array(trainingData, testData) = inputDf.randomSplit(Array(0.9, 0.1))

    // Train a DecisionTree model.
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()

    // Chain indexers and tree in a Pipeline.
    val pipelineDt: Array[PipelineStage] =
      Array(
        labelIndexer,
        dayOfWeekIndexer,
        pdDistrictIndexer,
        assembler,
        featureIndexer,
        dt,
        labelConverter)

    val pipelineRf: Array[PipelineStage] =
      Array(
        labelIndexer,
        dayOfWeekIndexer,
        pdDistrictIndexer,
        assembler,
        featureIndexer,
        rf,
        labelConverter)


    // Model selection
    val paramGrid = new ParamGridBuilder()
      .addGrid[Array[PipelineStage]](pipeline.stages, Array(pipelineDt, pipelineRf))
      .addGrid(dt.maxDepth, Array(4))
      .addGrid(rf.maxBins, Array(64))
      .addGrid(rf.maxDepth, Array(5, 10))
      .addGrid(rf.numTrees, Array(15, 30))
      .build()

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3) // Use 3+ in practice
      .setParallelism(5) // Evaluate up to 2 parameter settings in parallel

    // Train model. This also runs the indexers.
    val cvModel = cv.fit(trainingData)

    // Make predictions.
    val predictions = cvModel.transform(testData)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(100)


    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1.0 - accuracy}")

    println(s"Cross validation:")
    cvModel.avgMetrics.foreach(x => println(s"\t$x"))

    val bestModel = cvModel
      .bestModel
      .asInstanceOf[PipelineModel]
      .stages(5)

    bestModel match {
      case model: DecisionTreeClassificationModel =>
        println(s"DecisionTree classification tree model: $model")

      case model: RandomForestClassificationModel =>
        println(
          s"""RandomForest classification tree model:
             | numTrees: ${model.getNumTrees}
             | totalNumNodes: ${model.totalNumNodes}
             | maxDepth: ${model.getMaxDepth}
             | maxBins: ${model.getMaxBins}""".stripMargin
        )
    }

  }
}
