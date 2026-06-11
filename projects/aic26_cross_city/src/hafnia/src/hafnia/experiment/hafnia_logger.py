import importlib.util
import json
import os
import platform
import sys
import textwrap
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, field_validator

from hafnia.log import sys_logger, user_logger
from hafnia.utils import is_hafnia_cloud_job, now_as_str

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    user_logger.warning("MLFlow is not available")
    MLFLOW_AVAILABLE = False

SYSTEM_METRICS_AVAILABLE = importlib.util.find_spec("psutil") is not None
if not SYSTEM_METRICS_AVAILABLE:
    user_logger.warning("System monitoring is not available. Install 'psutil'")


class EntityType(Enum):
    """Types of entities that can be logged."""

    SCALAR = "scalar"
    METRIC = "metric"


class Entity(BaseModel):
    """
    Entity model for experiment logging.

    Attributes:
        step: Current step/iteration of the experiment
        ts: Timestamp when the entity was created
        name: Name of the entity
        ent_type: Type of entity (scalar, metric)
        mode: Mode of logging (train or eval)
        value: Numerical value of the entity
    """

    step: int
    ts: str
    name: str
    ent_type: str
    value: float = -1

    @field_validator("value", mode="before")
    def set_value(cls, v: Union[float, str, int]) -> float:
        """Convert input to float or default to -1.0."""
        try:
            return float(v)
        except (ValueError, TypeError) as e:
            user_logger.warning(f"Invalid value '{v}' provided, defaulting to -1.0: {e}")
            return -1.0

    @field_validator("ent_type", mode="before")
    def set_ent_type(cls, v: Union[EntityType, str]) -> str:
        """Convert EntityType enum to string value."""
        if isinstance(v, EntityType):
            return v.value
        return str(v)

    @staticmethod
    def create_schema() -> pa.Schema:
        """Create the PyArrow schema for the Parquet file."""
        return pa.schema(
            [
                pa.field("step", pa.int64()),
                pa.field("ts", pa.string()),
                pa.field("name", pa.string()),
                pa.field("ent_type", pa.string()),
                pa.field("value", pa.float64()),
            ]
        )


class HafniaLogger:
    EXPERIMENT_FILE = "experiment.parquet"

    def __init__(self, project_name: str, log_dir: Union[Path, str] = "./.data"):
        self._local_experiment_path = Path(log_dir) / "experiments" / now_as_str()
        self.project_name = project_name
        create_paths = [
            self._local_experiment_path,
            self.path_model_checkpoints(),
            self._path_artifacts(),
            self.path_model(),
        ]
        for path in create_paths:
            path.mkdir(parents=True, exist_ok=True)

        path_file = self.path_model() / "HOW_TO_STORE_YOUR_MODEL.txt"
        path_file.write_text(get_instructions_how_to_store_model())

        self.dataset_name: Optional[str] = None
        self.log_file = self._path_artifacts() / self.EXPERIMENT_FILE
        self.schema = Entity.create_schema()

        # Initialize MLflow for remote jobs
        self._mlflow_initialized = False
        if is_hafnia_cloud_job() and MLFLOW_AVAILABLE:
            self._init_mlflow()

        self.log_environment()
        self.log_configuration({"project_name": project_name})

    def _init_mlflow(self):
        """Initialize MLflow tracking for remote jobs."""
        try:
            # Set MLflow tracking URI from environment variable
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                user_logger.info(f"MLflow tracking URI set to: {tracking_uri}")

            # Set experiment name from environment variable
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
            if experiment_name:
                mlflow.set_experiment(experiment_name)
                user_logger.info(f"MLflow experiment set to: {experiment_name}")

            # Start MLflow run with tags
            run_name = os.getenv("MLFLOW_RUN_NAME", "undefined")
            created_by = os.getenv("MLFLOW_CREATED_BY")
            tags = {"project_name": self.project_name}
            if experiment_name:
                tags["organization_id"] = experiment_name
            if created_by:
                tags["created_by"] = created_by

            mlflow.start_run(run_name=run_name, tags=tags, log_system_metrics=SYSTEM_METRICS_AVAILABLE)
            self._mlflow_initialized = True
            user_logger.info("MLflow run started successfully")

        except Exception as e:
            user_logger.error(f"Failed to initialize MLflow: {e}")

    def path_local_experiment(self) -> Path:
        """Get the path for local experiment."""
        if is_hafnia_cloud_job():
            raise RuntimeError("Cannot access local experiment path in remote job.")
        return self._local_experiment_path

    def path_model_checkpoints(self) -> Path:
        """Get the path for model checkpoints."""
        if "MDI_CHECKPOINT_DIR" in os.environ:
            return Path(os.environ["MDI_CHECKPOINT_DIR"])

        if is_hafnia_cloud_job():
            return Path("/opt/ml/checkpoints")
        return self.path_local_experiment() / "checkpoints"

    def _path_artifacts(self) -> Path:
        """Get the path for artifacts."""
        if "MDI_ARTIFACT_DIR" in os.environ:
            return Path(os.environ["MDI_ARTIFACT_DIR"])

        if is_hafnia_cloud_job():
            return Path("/opt/ml/output/data")

        return self.path_local_experiment() / "data"

    def path_model(self) -> Path:
        """Get the path for the model."""
        if "MDI_MODEL_DIR" in os.environ:
            return Path(os.environ["MDI_MODEL_DIR"])

        if is_hafnia_cloud_job():
            return Path("/opt/ml/model")

        return self.path_local_experiment() / "model"

    def log_metric(self, name: str, value: float, step: int) -> None:
        self.log_scalar(name, value, step, EntityType.METRIC)

    def log_scalar(
        self,
        name: str,
        value: float,
        step: int,
        ent_type: EntityType = EntityType.SCALAR,
    ) -> None:
        entity = Entity(
            step=step,
            ts=datetime.now().isoformat(),
            name=name,
            value=value,
            ent_type=ent_type.value,
        )
        self.write_entity(entity)

        # Also log to MLflow if initialized
        if not self._mlflow_initialized:
            return
        try:
            mlflow.log_metric(name, value, step=step)
        except Exception as e:
            user_logger.error(f"Failed to log metric to MLflow: {e}")

    def log_configuration(self, configurations: Dict):
        self.log_hparams(configurations, "configuration.json")

    def log_hparams(self, params: Dict, fname: str = "hparams.json"):
        file_path = self._path_artifacts() / fname
        try:
            if file_path.exists():  # New params are appended to existing params
                existing_params = json.loads(file_path.read_text())
            else:
                existing_params = {}
            existing_params.update(params)
            file_path.write_text(json.dumps(existing_params, indent=2))
            user_logger.info(f"Saved parameters to {file_path}")

            # Also log to MLflow if initialized
            if not self._mlflow_initialized:
                return
            try:
                mlflow.log_params(params)
            except Exception as e:
                user_logger.error(f"Failed to log params to MLflow: {e}")

        except Exception as e:
            user_logger.error(f"Failed to save parameters to {file_path}: {e}")

    def log_environment(self):
        environment_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "os": platform.system(),
            "os_release": platform.release(),
            "cpu_count": os.cpu_count(),
            "cuda_version": os.getenv("CUDA_VERSION", "N/A"),
            "cudnn_version": os.getenv("CUDNN_VERSION", "N/A"),
        }
        self.log_hparams(environment_info, "environment.json")

    def write_entity(self, entity: Entity) -> None:
        """
        Force writing all accumulated logs to disk.

        This method should be called at the end of an experiment or
        when you want to ensure all logs are written.
        """
        print(entity)  # Keep this line! Parsed and used for real-time logging in another process
        entities = [entity]
        try:
            log_batch = pa.Table.from_pylist([e.model_dump() for e in entities], schema=self.schema)

            if not self.log_file.exists():
                pq.write_table(log_batch, self.log_file)
            else:
                prev = pa.parquet.read_table(self.log_file)
                next_table = pa.concat_tables([prev, log_batch])
                pq.write_table(next_table, self.log_file)
        except Exception as e:
            sys_logger.error(f"Failed to flush logs: {e}")

    def end_run(self) -> None:
        """End the MLflow run if initialized."""
        if not self._mlflow_initialized:
            return
        try:
            mlflow.end_run()
            self._mlflow_initialized = False
            user_logger.info("MLflow run ended successfully")
        except Exception as e:
            user_logger.error(f"Failed to end MLflow run: {e}")

    def __del__(self):
        """Cleanup when logger is destroyed."""
        self.end_run()


def get_instructions_how_to_store_model() -> str:
    instructions = textwrap.dedent(
        """\
        If you, against your expectations, don't see any models in this folder,
        we have provided a small guide to help.

        The hafnia TaaS framework expects models to be stored in a folder generated
        by the hafnia logger. You will need to store models in this folder
        to  ensure that they are properly stored and accessible after training.

        Please check your recipe script and ensure that the models are being stored
        as expected by the TaaS framework.

        Below is also a small example to demonstrate:

        ```python
        from hafnia.experiment import HafniaLogger

        # Initiate Hafnia logger
        logger = HafniaLogger(project_name="my_classification_project")

        # Folder path to store models - generated by the hafnia logger. 
        model_dir = logger.path_model()

        # Example for storing a pytorch based model. Note: the model is stored in 'model_dir'
        path_pytorch_model = model_dir / "model.pth"

        # Finally save the model to the specified path
        torch.save(model.state_dict(), path_pytorch_model)
        ```
        """
    )

    return instructions
