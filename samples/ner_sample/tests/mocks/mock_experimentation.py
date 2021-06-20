from ner_sample.experimentation import Experimentation


class MockExperimentation(Experimentation):
    def __init__(self):
        self.params = {}
        self.metrics = {}
        super().__init__()

    def set_experiment(self, name, artifact_location=None):
        pass

    def start_run(self):
        pass

    def end_run(self):
        pass

    def log_param(self, key, value):
        self.params[key] = value

    def log_params(self, params):
        self.params.update(params)

    def log_metric(self, key, value, step=None):
        self.metrics[key] = value

    def log_metrics(self, metrics, step=None):
        self.metrics.update(metrics)

    def log_image(self, title, fig):
        pass

    def log_artifact(self, local_path, name=None, artifact_path=None):
        pass

    def log_artifacts(self, local_path, name=None, artifact_path=None):
        pass
