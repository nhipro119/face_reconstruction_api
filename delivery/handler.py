from . import api
class Handler:
    def __init__(self) -> None:
        self.api = api.Api()
    def setup(self,router):
        router.add_url_rule("/predict","predict",self.api.predict, methods=["POST"])
        router.add_url_rule("/reconstructed-face","reconstructed-face",self.api.get_reconstruction_file, methods=["GET"])
        router.add_url_rule("/split-wound","split-wound",self.api.get_split_file, methods=["GET"])
        return router