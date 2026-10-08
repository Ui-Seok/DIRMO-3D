from .flowr import CapabilityError


class LiFTComparison:
    def capabilities(self):
        return {"fixed_graph": False, "semantic_joint_generation": False,
                "status": "disabled_comparison_placeholder",
                "reason": "LiFT needs DrugFlow/SMI-TED semantic conditioning; exact graph is not guaranteed"}

    def generate_fixed_graph(self, request):
        raise CapabilityError("LiFT is a semantic joint comparison, not a fixed-graph generator")

    def generate_local_fragment(self, request):
        raise CapabilityError("LiFT comparison is not implemented in version 0.1")

    def health(self):
        return self.capabilities()
