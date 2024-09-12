import sys
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    try:
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
        logging.info("Training pipleline execution completed successfully.")

    except Exception as e:
        raise SensorException(e, sys)