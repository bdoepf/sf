package com.github.bdoepf.sf

import org.apache.spark.ml.{PipelineStage, Transformer}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.tuning.ParamGridBuilder


class LrModel(labelColName: String = "indexedLabel", featureColName: String = "features") extends SfModel {

  private val lr = new LogisticRegression()
    .setFamily("multinomial")
    .setLabelCol(labelColName).setFeaturesCol(featureColName)

  override def model(): PipelineStage = lr

  override def addGridParams(builder: ParamGridBuilder): ParamGridBuilder = {
    builder.addGrid(lr.maxIter, Array(500))
      .addGrid(lr.regParam, Array(0.01))
      .addGrid(lr.elasticNetParam, Array(0.8))
  }

  override def evaluate(bestModel: Transformer): Unit = {
    bestModel match {
      case model: LogisticRegressionModel =>
        model.save(s"model/LogisticRegressionModel_${System.currentTimeMillis()}")
        println(
          s"""LogisticRegression classification :
             | maxIter: ${model.getMaxIter}
             | regParam: ${model.getRegParam}
             | elasticNetParam: ${model.getElasticNetParam}""".stripMargin
        )
        println("Accuracy of summary" + model.summary.accuracy)
      case _ => println("Best model isn't a LogisticRegressionModel.")
    }
  }
}

