from pathlib import Path
import ast
import types
import unittest


class VisibilityGateTest(unittest.TestCase):
    def setUp(self):
        source = (Path(__file__).resolve().parents[1] / 'scripts/export_texverse_clip.py').read_text(encoding='utf-8')
        outer = ast.parse(source)
        function = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == 'blender_export_script')
        template = next(n.value.value for n in function.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
        inner = ast.parse(template)
        guard = next(n for n in inner.body if isinstance(n,ast.FunctionDef) and n.name=='reject_invisible_sampled_mesh')
        def fail(reason, **kwargs):
            raise ValueError(reason)
        namespace = {'fail':fail}
        exec(compile(ast.Module(body=[guard],type_ignores=[]),'<visibility_guard>','exec'), namespace)
        self.guard = namespace['reject_invisible_sampled_mesh']

    def test_visible_mesh_is_unchanged(self):
        obj = types.SimpleNamespace(name='body',hide_viewport=False,hide_render=False,visible_get=lambda:True)
        self.assertIsNone(self.guard(obj,1))

    def test_hidden_mesh_is_rejected_for_each_visibility_route(self):
        for viewport, render, visible in [(True,False,True),(False,True,True),(False,False,False)]:
            obj=types.SimpleNamespace(name='body',hide_viewport=viewport,hide_render=render,visible_get=lambda:visible)
            with self.assertRaisesRegex(ValueError,'selected_mesh_hidden_at_sampled_frame'):
                self.guard(obj,1)


if __name__=='__main__':
    unittest.main()
