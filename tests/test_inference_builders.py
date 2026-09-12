"""Model builders must not implicitly opt dynamic audio into CUDA graphs."""
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class BuilderTests(unittest.TestCase):
    def test_default_models_use_eager_without_compilation(self):
        for name, symbol in [('cli.inference_svc', 'SoulXSingerSVC'), ('cli.inference', 'SoulXSinger')]:
            module = importlib.import_module(name)
            model = Mock()
            model.to.return_value = model
            model.parameters.return_value = []
            estimator = object()
            model.cfm_decoder.model.diff_estimator = estimator
            with self.subTest(name=name), patch.object(module.os.path, 'isfile', return_value=True), patch.object(module, symbol, return_value=model), patch.object(module.torch, 'load', return_value={'state_dict': {}}), patch.object(module.torch, 'compile') as compile_model:
                actual = module.build_model('fixture.pt', SimpleNamespace(), device='cpu')
                compile_model.assert_not_called()
                self.assertIs(actual, model)
                self.assertIs(model.cfm_decoder.model.diff_estimator, estimator)
                model.eval.assert_called_once()
                model.load_state_dict.assert_called_once_with({}, strict=True)

    def test_explicit_compilation_avoids_cuda_graphs(self):
        for name, symbol in [('cli.inference_svc', 'SoulXSingerSVC'), ('cli.inference', 'SoulXSinger')]:
            module = importlib.import_module(name)
            model = Mock()
            model.to.return_value = model
            model.parameters.return_value = []
            estimator, compiled = object(), object()
            model.cfm_decoder.model.diff_estimator = estimator
            with self.subTest(name=name), patch.object(module.os.path, 'isfile', return_value=True), patch.object(module, symbol, return_value=model), patch.object(module.torch, 'load', return_value={'state_dict': {}}), patch.object(module.torch, 'compile', return_value=compiled) as compile_model:
                module.build_model('fixture.pt', SimpleNamespace(), device='cpu', compile_model=True)
                compile_model.assert_called_once_with(estimator, options={'triton.cudagraphs': False})
                self.assertIs(model.cfm_decoder.model.diff_estimator, compiled)


if __name__ == '__main__':
    unittest.main()
