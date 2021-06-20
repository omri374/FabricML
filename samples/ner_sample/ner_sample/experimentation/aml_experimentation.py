import logging

try:
    from azureml.core import Workspace, Experiment
except ImportError:
    pass

from ner_sample.experimentation import Experimentation


class AmlExperimentation(Experimentation):

    def __init__(self, ws):
        super().__init__()
        self.aml_ws = ws
        self.aml_experiment = None
        self.aml_run = None
        self.is_running_flag = False

    def set_experiment(self, name, artifact_location=None):
        logging.info("Connecting to Azure ML")
        self.aml_experiment = Experiment(workspace=self.aml_ws, name=name)

    def start_run(self):
        self.aml_run = self.aml_experiment.start_logging()
        self.is_running_flag = True

    def end_run(self):
        self.aml_run.complete()
        self.is_running_flag = False

    def log_param(self, key, value):
        self.aml_run.log(key, value)

    def log_params(self, params):
        self.aml_run.log(params)

    def log_metric(self, key, value, step=None):
        self.aml_run.log(key, value)

    def log_metrics(self, metrics, step=None):
        self.aml_run.log(metrics)

    def search_runs(
        self,
        experiment_ids=None,
        filter_string="",
        run_view_type=1,
        max_results=100000,
        order_by=None,
    ):
        raise NotImplementedError()

    def log_image(self, title, fig):
        self.aml_run.log_image(name=title, plot=fig)

    def log_artifact(self, local_path, name=None, artifact_path=None):
        self.aml_run

    def log_artifacts(self, local_path, name=None, artifact_path=None):
        pass
