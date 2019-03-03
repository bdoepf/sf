package com.github.bdoepf.sf.auto

import com.salesforce.op.{OpParams, OpWorkflow, OpWorkflowRunner}
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features._
import com.salesforce.op.features.types.{DateTime, _}
import com.salesforce.op.readers._
import com.salesforce.op.stages.impl.classification.MultiClassificationModelSelector
import com.salesforce.op.stages.impl.evaluator.LogLoss
import com.salesforce.op.stages.impl.tuning.DataCutter
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{hour, month, to_timestamp}
import org.apache.spark.sql.types.{DoubleType, IntegerType}

// Spark enrichments (optional)

case class Item(Dates: String, Category: String, DayOfWeek: String, PdDistrict: String, /*Address: String,*/ X: Double, Y: Double)

object SFAppAutoML {
  def main(args: Array[String]): Unit = {
    val randomSeed = 1644323
    implicit val spark = SparkSession.builder.master("local[*]").getOrCreate()
    import spark.implicits._

    val format = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss")

    val dateTime = FeatureBuilder.DateTime[Item].extract(x => new DateTime(format.parse(x.Dates).getTime)).asPredictor
    val dayOfWeek = FeatureBuilder.PickList[Item].extract(x => Option(x.PdDistrict).toPickList).asPredictor
    val pdDistrict = FeatureBuilder.PickList[Item].extract(x => Option(x.PdDistrict).toPickList).asPredictor
    // val address = FeatureBuilder.Text[Item].extract(x => Option(x.Address).toText).asPredictor
    val pos = FeatureBuilder.Geolocation[Item]("position").extract(item => new Geolocation(lat = item.Y, lon = item.X, accuracy = GeolocationAccuracy.Unknown)).asPredictor

    val category = FeatureBuilder.PickList[Item].extract(x => Option(x.Category).toPickList).asResponse

    val labels = category.indexed()
    val features = Seq(dateTime, dayOfWeek, pdDistrict,/* address, */pos).transmogrify()

    // Read data as a DataFrame
//   val allTrainingsData = DataReaders.Simple.csvCase[Item](path = Some("/home/user1/Projects/SF/data/train-without-header.csv")).readDataset().toDF()
   val allTrainingsData = spark
     .read
     .option("header", "true")
     .csv("data/train.csv")
     .drop("Descript", "Resolution", "Address")
     .withColumn("X", $"X".cast(DoubleType))
     .withColumn("Y", $"Y".cast(DoubleType))
     .as[Item]
     .limit(10000) // TODO REMOVE

    allTrainingsData.show(false)


    val pred = MultiClassificationModelSelector
      .withCrossValidation(splitter = Some(DataCutter(reserveTestFraction = 0.2, seed = randomSeed)), seed = randomSeed)
      .setInput(labels, features).getOutput()

    val evaluator = LogLoss.multiLogLoss //Evaluators.MultiClassification.f1()
      .setLabelCol(labels)
      .setPredictionCol(pred)

    val model = new OpWorkflow().setInputDataset(allTrainingsData).setResultFeatures(pred, labels).train()

    println("Model summary:\n" + model.summaryPretty())
//
//    model.
//    evaluator.evaluateAll(allTrainingsData)

//    def runner(opParams: OpParams): OpWorkflowRunner =
//      new OpWorkflowRunner(
//        workflow = wf,
//        trainingReader = reader,
//        scoringReader = reader,
//        evaluationReader = Option(irisReader),
//        evaluator = Option(evaluator),
//        featureToComputeUpTo = Option(features)
//      )

    // Manifest the result features of the workflow
    println("Scoring the model")
    val (dataframe, metrics) = model.scoreAndEvaluate(evaluator = evaluator)

    println("Transformed dataframe columns:")
    dataframe.columns.foreach(println)
    println("Metrics:")
    println(metrics)

  }
}
