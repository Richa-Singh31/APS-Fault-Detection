import os
import sys
import pandas as pd
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity.config_entity import ModelEvaluationConfig
from sensor.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataValidationArtifact
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel, TargetValueMapping, ModelResolver
from sensor.utils.main_utils import save_object, load_object, write_yaml_file
from sensor.constant.training_pipeline import TARGET_COLUMN

class ModelEvaluation:
    
    def __init__(self, model_eval_config: ModelEvaluationConfig, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ModelTrainerArtifact) -> None:
        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            df = pd.concat([train_df, test_df])
            y_true = df[TARGET_COLUMN]
            y_true.replace(TargetValueMapping().mapping(), inplace=True)
            df.drop(TARGET_COLUMN, inplace=True, axis=1)

            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted = True

            if not model_resolver.model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, improved_accuracy=None, best_model_path=None, trained_model_path=train_model_file_path, trained_model_metric_artifact=self.model_trainer_artifact.train_metric_artifact, latest_model_metric_artifact=None
                )

                logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
        
            latest_model_path = model_resolver.get_best_model_path()
            logging.info(f"latest model path: {latest_model_path}")
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)

            y_train_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)
            logging.info(f"y latest path: {y_latest_pred}")

            train_metric = get_classification_score(y_true, y_train_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)
            logging.info(f"latest metric: {latest_metric}")

            improved_accuracy = train_metric.f1_score - latest_metric.f1_score
            logging.info(f"improved accuracy: {improved_accuracy}")
            if self.model_eval_config.changed_threshold < improved_accuracy:
                is_model_accepted = True
            else:
                is_model_accepted = False

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path, trained_model_path=train_model_file_path, trained_model_metric_artifact=train_metric, latest_model_metric_artifact=latest_metric
            )

            model_eval_report = model_evaluation_artifact.__dict__
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)
            logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
    
        except Exception as e:
            raise SensorException(e, sys)