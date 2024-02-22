class Model_list:

    model_list = []

    def __init__(self, modelList):
        self.model_list = modelList

    def get_model_names_list(self):
        model_names = []
        for model in self.model_list:
            model_names.append(model.name)
        return model_names

    def append(self, model):
        self.model_list.append(model)

    def get_model_by_name(self, model_name):
        for model in self.model_list:
            if model.name == model_name:
                return model

