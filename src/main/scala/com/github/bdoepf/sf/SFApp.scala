package com.github.bdoepf.sf

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage, linalg}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.slf4j.LoggerFactory

object SFApp {

  def main(args: Array[String]): Unit = {
    val log = LoggerFactory.getLogger(this.getClass.getName.stripSuffix("$"))
    setLogLevels()

    implicit val spark: SparkSession = SparkSession
      .builder()
      .config(new SparkConf()
        .set("spark.local.dir", s"${sys.env("HOME")}/spark-tmp"))
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    val inputDf = spark
      .read
      .option("header", "true")
      .csv("data/train.csv")
      .drop("Descript", "Resolution", "Address")
      .withColumnRenamed("Category", "label")
      .withColumn("X", $"X".cast(DoubleType))
      .withColumn("Y", $"Y".cast(DoubleType))
      .withColumn("hour", hour(to_timestamp($"Dates")))
      .withColumn("month", month(to_timestamp($"Dates")))
      .drop("Dates")
    // .sample(0.01) // for testing take only a fraction of all input data

    inputDf.printSchema()
    inputDf.cache()
    // Split the data into training and test sets
    val Array(trainingData, testData) = inputDf.randomSplit(Array(0.9, 0.1))

    // Label, crime categories
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(inputDf)

    val numberOfCrimeCategories = labelIndexer.labels.length
    log.info("labelIndexer: " + numberOfCrimeCategories)

    val dayOfWeekIndexer = new StringIndexer()
      .setInputCol("DayOfWeek")
      .setOutputCol("indexedDayOfWeek")
      .fit(inputDf)

    log.info("dayOfWeekIndexer: " + dayOfWeekIndexer.labels.length)

    val pdDistrictIndexer = new StringIndexer()
      .setInputCol("PdDistrict")
      .setOutputCol("indexedPdDistrict")
      .fit(inputDf)

    log.info("pdDistrictIndexer: " + pdDistrictIndexer.labels.length)

    val encoderOutputCols = Array(s"${dayOfWeekIndexer.getOutputCol}Vec", s"${pdDistrictIndexer.getOutputCol}Vec", "hourVec", "monthVec" /*, s"${addressIndexer.getOutputCol}Vec}"*/)
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array(dayOfWeekIndexer.getOutputCol, pdDistrictIndexer.getOutputCol, /*addressIndexer.getOutputCol,*/ "hour", "month"))
      .setOutputCols(encoderOutputCols)

    val xyAssembler = new VectorAssembler()
      .setInputCols(Array("X", "Y"))
      .setOutputCol("position")

    val scaler = new MinMaxScaler()
      .setInputCol("position")
      .setOutputCol("positionScaled")

    val finalAssembler = new VectorAssembler()
      .setInputCols("positionScaled" +: encoderOutputCols)
      .setOutputCol("features")

    // Following transformers are required for Random Forest
    val assembler = new VectorAssembler()
      .setInputCols(Array("indexedDayOfWeek", "indexedPdDistrict", "X", "Y", "hour", "month"))
      .setOutputCol("remainingFeatures")

    // Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer()
      .setInputCol("remainingFeatures")
      .setOutputCol("features")
      .setMaxCategories(40) // features with > distinct values are treated as continuous.


    //    val lr = new LrModel()
    //    val lrModel = lr.model()
    //    log.info(lrModel.explainParams())
    //    val lrStages: Array[PipelineStage] = Array(
    //      labelIndexer,
    //      dayOfWeekIndexer,
    //      pdDistrictIndexer,
    //      encoder,
    //      xyAssembler,
    //      scaler,
    //      finalAssembler,
    //      lrModel)

    // Random Forest
    val rf = new RfModel()
    val rfModel = rf.model()
    val pipelineRf: Array[PipelineStage] =
      Array(
        labelIndexer,
        dayOfWeekIndexer,
        pdDistrictIndexer,
        assembler,
        featureIndexer,
        rfModel)


    // MultilayerPerceptron
    val mpcPrepStages: Array[PipelineStage] = Array(
      labelIndexer,
      dayOfWeekIndexer,
      pdDistrictIndexer,
      encoder,
      xyAssembler,
      scaler,
      finalAssembler)
    val model = new Pipeline().setStages(mpcPrepStages).fit(inputDf).transform(inputDf)
    val numOfinputFeatures = model.schema("features").metadata.getMetadata("ml_attr").getLong("num_attrs")
    log.info("NumberOfInputFeatures :" + numOfinputFeatures)
    val mlp = new MLPModel(numOfinputFeatures.toInt, numberOfCrimeCategories)
    val mlpModel = mlp.model()
    mlpModel.explainParams()
    val mpcStages: Array[PipelineStage] = Array(
      labelIndexer,
      dayOfWeekIndexer,
      pdDistrictIndexer,
      encoder,
      xyAssembler,
      scaler,
      finalAssembler,
      mlpModel)

    // ML Pipeline
    val models = List(rf, mlp)
    val pipeline = new Pipeline()
    val baseGridBuilder = new ParamGridBuilder()
      .addGrid[Array[PipelineStage]](pipeline.stages, Array(/*lrStages, */ pipelineRf, mpcStages))

    // add hyper params for tuning
    val paramGrid = models
      .foldLeft(baseGridBuilder)((grid: ParamGridBuilder, model: SfModel) => model.addGridParams(grid))
      .build

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3) // Use 3+ in practice
      .setParallelism(5) // Evaluate in parallel

    // Train model. This also runs the indexers
    val cvModel = cv.fit(trainingData)
    // Make predictions
    val predictions = cvModel.transform(testData)

    // Check which model was the best and save it
    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages.last
    models.foreach(m => m.evaluate(bestModel))

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Select example rows to display.
    labelConverter.transform(predictions).select("predictedLabel", "label").show(100, truncate = false)

    val accuracy = evaluator.evaluate(predictions)
    log.info(s"Test Error = ${1.0 - accuracy}")

    // SUBMISSION Id,Dates,DayOfWeek,PdDistrict,Address,X,Y
    val testDf = spark
      .read
      .option("header", "true")
      .csv("data/test.csv")
      .withColumn("Id", $"Id".cast(IntegerType))
      .withColumn("X", $"X".cast(DoubleType))
      .withColumn("Y", $"Y".cast(DoubleType))
      .withColumn("hour", hour(to_timestamp($"Dates")))
      .withColumn("month", month(to_timestamp($"Dates")))
      .drop("Dates", "Address")

    val testPredictions = cvModel.transform(testDf)

    // The column containing probabilities has to be converted from Vector to Array
    val vecToArray = udf((xs: linalg.Vector) => xs.toArray)
    val dfArr = testPredictions.withColumn("probabilityArr", vecToArray($"probability"))

    val probColumns = labelIndexer.labels.zipWithIndex.map {
      case (alias, idx) => (alias, col("probabilityArr").getItem(idx).as(alias))
    }

    val columnsAdded = probColumns.foldLeft(dfArr) { case (d, (colName, colContents)) =>
      if (d.columns.contains(colName)) {
        d
      } else {
        d.withColumn(colName, colContents)
      }
    }
    val resultDf = columnsAdded.select("id", labelIndexer.labels.toList: _*)
    resultDf.show()
    resultDf.repartition(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", value = true)
      .option("compression", "gzip")
      .csv("data/submission/")
  }

  private def setLogLevels(): Unit = {
    Logger.getRootLogger.setLevel(Level.WARN)
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("com.github.bdoepf.sf").setLevel(Level.INFO)
    val interestingSparkLoggers = List("org.apache.spark.ml.util.Instrumentation", "org.apache.spark.mllib.optimization.GradientDescent")
    interestingSparkLoggers.foreach(logName => Logger.getLogger(logName.stripSuffix("$")).setLevel(Level.INFO))
  }
}
