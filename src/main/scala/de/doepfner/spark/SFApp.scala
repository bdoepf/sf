package de.doepfner.spark

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{SaveMode, SparkSession}


object SFApp {

  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession.builder().config(new SparkConf().set("spark.local.dir", "/home/user1/spark-tmp")).master("local[*]").getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("WARN")

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

    inputDf.cache()
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = inputDf.randomSplit(Array(0.9, 0.1))

    // Label, crime categories
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(inputDf)

    println("labelIndexer: " + labelIndexer.labels.length)

    val dayOfWeekIndexer = new StringIndexer()
      .setInputCol("DayOfWeek")
      .setOutputCol("indexedDayOfWeek")
      .fit(inputDf)

    println("dayOfWeekIndexer: " + dayOfWeekIndexer.labels.length)

    val pdDistrictIndexer = new StringIndexer()
      .setInputCol("PdDistrict")
      .setOutputCol("indexedPdDistrict")
      .fit(inputDf)

    println("pdDistrictIndexer: " + pdDistrictIndexer.labels.length)


    // START EXAMPLE

    //    val sentenceData = spark.createDataFrame(Seq(
    //      (0.0, "Hi I heard about Spark"),
    //      (0.0, "I wish Java could use case classes"),
    //      (1.0, "Logistic regression models are neat")
    //    )).toDF("label", "sentence")

    //    val tokenizer = new Tokenizer().setInputCol("Address").setOutputCol("addressTokenized")
    //
    //    val remover = new StopWordsRemover()
    //      .setInputCol("addressTokenized")
    //      .setOutputCol("addressTokenizedFiltered")

    //    remover.transform(tokenizer.transform(inputDf)).show(100, false)
    //    sys.exit(0)

    //    val hashingTF = new HashingTF()
    //      .setInputCol("addressTokenizedFiltered").setOutputCol("addressHashed").setNumFeatures(5000)
    //
    //    val idf = new IDF().setInputCol("addressHashed").setOutputCol("addressHashedIdf")
    // END EXAMPLE

    /*    val addressIndexer = new StringIndexer()
          .setInputCol("Address")
          .setOutputCol("indexedAddress")
          .fit(inputDf)

        println("addressIndexer: " + addressIndexer.labels.length)
    */
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


    val lr = new LrModel()
    val lrModel = lr.model()
    println(lrModel.explainParams())
    val lrStages: Array[PipelineStage] = Array(
      labelIndexer,
      dayOfWeekIndexer,
      pdDistrictIndexer,
      encoder,
      xyAssembler,
      scaler,
      finalAssembler,
      lrModel)

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

    val pipeline = new Pipeline()
    val gridBuilder = new ParamGridBuilder()
      .addGrid[Array[PipelineStage]](pipeline.stages, Array(lrStages, pipelineRf))

    // add lr params
    val withLrParams = lr.addGridParams(gridBuilder)

    // add rf params
    val withRfParams = rf.addGridParams(withLrParams)

    // hyper params
    val paramGrid = withRfParams.build()

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
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

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages.last //.asInstanceOf[LogisticRegressionModel]
    lr.evaluate(bestModel)
    rf.evaluate(bestModel)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Select example rows to display.
    labelConverter.transform(predictions).select("predictedLabel", "label").show(100, false)

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1.0 - accuracy}")

    // TODO REFACTORING SUBMISSION
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
    val testPredictionsRelabled = labelConverter.transform(testPredictions).select("Id", "predictedLabel")
    testPredictionsRelabled.show(100, false)
    val resultDf = testPredictionsRelabled
      .as[ResultItem]
      .map(Result.convertToSubmission)

    resultDf.cache()
    val count = resultDf.count()
    println("Count: " + count)
    assert(count == 884262)
    resultDf
      .repartition(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", true)
      .option("compression", "gzip")
      .csv("data/submission/")


    /*
        val assembler = new VectorAssembler()
          .setInputCols(Array("indexedDayOfWeek", "indexedPdDistrict", "X", "Y", "hour", "month"))
          .setOutputCol("remainingFeatures")



        val assembler2 = new VectorAssembler()
          .setInputCols(Array("remainingIndexedFeatures", "addressHashedIdf"))
          .setOutputCol("indexedFeatures")
        // .fit(data)



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
              tokenizer,
              remover,
              hashingTF,
              idf,
              assembler,
              featureIndexer,
              assembler2,
              dt,
              labelConverter)

          val pipelineRf: Array[PipelineStage] =
            Array(
              labelIndexer,
              dayOfWeekIndexer,
              pdDistrictIndexer,
              tokenizer,
              remover,
              hashingTF,
              idf,
              assembler,
              featureIndexer,
              assembler2,
              rf,
              labelConverter)


          // Model selection
          val paramGrid = new ParamGridBuilder()
            .addGrid[Array[PipelineStage]](pipeline.stages, Array(pipelineDt, pipelineRf))
            .addGrid(dt.maxDepth, Array(4))
            .addGrid(rf.maxBins, Array(32))
            .addGrid(rf.maxDepth, Array(15))
            .addGrid(rf.numTrees, Array(30))
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
          predictions.select("predictedLabel", "label").show(100, false)


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
      */
  }
}
