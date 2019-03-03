package com.github.bdoepf.sf

import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.{PipelineStage, Transformer}

trait SfModel {
  def model(): PipelineStage

  def addGridParams(builder: ParamGridBuilder): ParamGridBuilder

  def evaluate(transformer: Transformer): Unit
}



