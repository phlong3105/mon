from hafnia.experiment import HafniaLogger

batch_size = 128
learning_rate = 0.001

# Initialize Hafnia logger
logger = HafniaLogger(project_name="example_classification_project")

# Log experiment parameters
logger.log_configuration({"batch_size": 128, "learning_rate": 0.001})

# Store checkpoints in this path
ckpt_dir = logger.path_model_checkpoints()

# Store the trained model in this path
model_dir = logger.path_model()

# Log scalar and metric values during training and validation
logger.log_scalar("train/loss", value=0.1, step=100)
logger.log_metric("train/accuracy", value=0.98, step=100)

logger.log_scalar("validation/loss", value=0.1, step=100)
logger.log_metric("validation/accuracy", value=0.95, step=100)
