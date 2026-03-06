import shap

class RiskExplainer:
    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)

    def explain(self, x):
        return self.explainer.shap_values([x])
