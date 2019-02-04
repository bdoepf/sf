package de.doepfner.spark

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.{PipelineStage, Transformer}

trait SfModel {
  def model(): PipelineStage

  def addGridParams(builder: ParamGridBuilder): ParamGridBuilder

  def evaluate(transformer: Transformer): Unit
}


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


class RfModel(labelColName: String = "indexedLabel", featureColName: String = "features") extends SfModel {

  private val rf = new RandomForestClassifier()
    .setLabelCol(labelColName)
    .setFeaturesCol(featureColName)

  override def model(): PipelineStage = rf

  override def addGridParams(builder: ParamGridBuilder): ParamGridBuilder = {
    builder.addGrid(rf.numTrees, Array(30))
      .addGrid(rf.maxDepth, Array(5, 15))
      .addGrid(rf.maxBins, Array(40))
  }

  override def evaluate(bestModel: Transformer): Unit = {
    bestModel match {
      case model: RandomForestClassificationModel =>
        println(
          s"""RandomForest classification tree model:
             | numTrees: ${model.getNumTrees}
             | totalNumNodes: ${model.totalNumNodes}
             | maxDepth: ${model.getMaxDepth}
             | maxBins: ${model.getMaxBins}""".stripMargin
        )
      case _ => println("Best model isn't a RandomForestClassificationModel.")
    }
  }
}
