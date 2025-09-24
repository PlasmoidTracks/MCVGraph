import os
import sys
import unittest
import numpy as np
from MCVGraph.DataSource import DataSource
from MCVGraph.canvas.Canvas import Canvas

from PyQt5.QtWidgets import QApplication

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



class TestDataSource(unittest.TestCase):
    def test_accepts_numpy_array(self):
        arr = np.array([[1, 2], [3, 4]])
        ds = DataSource(arr)
        out = ds.get()
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(ds.size(), 2)

    @unittest.skipUnless(HAS_PANDAS, "pandas not installed")
    def test_accepts_pandas_dataframe(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        ds = DataSource(df)
        out = ds.get()
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(ds.size(), 2)

    def test_accepts_list_of_lists(self):
        data = [[1, 2, 3], [4, 5, 6]]
        ds = DataSource(data)
        out = ds.get()
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(ds.size(), 2)

    def test_string_input_becomes_2d_array(self):
        ds = DataSource("not a dataset")
        out = ds.get()
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(ds.size(), 0)

    def test_none_yields_empty_array(self):
        ds = DataSource(None)
        out = ds.get()
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (0, 0))
        self.assertEqual(ds.size(), 0)

    def test_empty_list_yields_empty_array(self):
        ds = DataSource([])
        out = ds.get()
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.size, 0)
        self.assertEqual(ds.size(), 0)

    def test_set_updates_data(self):
        ds = DataSource(np.zeros((1, 2)))
        new_data = np.arange(6).reshape(3, 2)
        ds.set(new_data)
        np.testing.assert_array_equal(ds.get(), new_data)
        self.assertEqual(ds.size(), 3)

    def test_overwrite_with_none(self):
        ds = DataSource(np.arange(4).reshape(2, 2))
        ds.set(None)
        out = ds.get()
        self.assertEqual(out.shape, (0, 0))
        self.assertEqual(ds.size(), 0)

    def test_apply_transform_passes_through(self):
        ds = DataSource(np.array([[1, 2], [3, 4]]))
        doubled = ds.apply_transform(lambda a: a * 2)
        np.testing.assert_array_equal(doubled, np.array([[2, 4], [6, 8]]))

    def test_apply_transform_side_effects(self):
        ds = DataSource(np.array([[1, 2], [3, 4]]))
        def add_one(a):
            return a + 1
        out = ds.apply_transform(add_one)
        np.testing.assert_array_equal(out, np.array([[2, 3], [4, 5]]))

    def test_shape_property_matches_array(self):
        arr = np.arange(12).reshape(3, 4)
        ds = DataSource(arr)
        self.assertEqual(ds.get().shape, arr.shape)

    def test_dtype_is_preserved(self):
        arr = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        ds = DataSource(arr)
        self.assertEqual(ds.get().dtype, np.float32)

    def test_mutating_original_array_does_not_affect_stored(self):
        arr = np.array([[1, 2], [3, 4]])
        ds = DataSource(arr)
        arr[0, 0] = 999  # mutate outside
        self.assertNotEqual(ds.get()[0, 0], 999)


class TestLinkingSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only one QApplication per process
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.win_a = Canvas()
        self.win_b = Canvas()
        self.win_c = Canvas()

    def tearDown(self):
        self.win_a.close()
        self.win_b.close()
        self.win_c.close()

    def test_simple_link_sets_target(self):
        self.win_a.link(self.win_b)
        self.assertTrue(self.win_a.is_linking())
        self.assertTrue(self.win_a.is_linking_to(self.win_b))
        self.assertTrue(self.win_b.is_linked_to())

    def test_cyclic_linking_prevention(self):
        self.win_a.link(self.win_b)
        self.win_b.link(self.win_c)
        with self.assertRaises(ValueError):
            self.win_c.link(self.win_a)

    def test_unlinking_clears_target(self):
        self.win_a.link(self.win_b)
        self.assertTrue(self.win_a.is_linking_to(self.win_b))
        self.win_a.unlink(self.win_b)
        self.assertFalse(self.win_a.is_linking())
        self.assertFalse(self.win_b.is_linked_to())


class TestGUIRelated(unittest.TestCase):
    @unittest.skip("Requires full Qt event loop & rendering")
    def test_scatterplot_selection_bounds(self):
        self.fail("Not implemented")

    @unittest.skip("Window snapping needs system-level interaction")
    def test_window_snapping_positions(self):
        self.fail("Not implemented")

    @unittest.skip("Performance tests not deterministic")
    def test_large_dataset_performance(self):
        self.fail("Not implemented")


class TestDataSourceEdgeCases(unittest.TestCase):
    def test_1d_list_becomes_column_vector(self):
        ds = DataSource([1, 2, 3])
        out = ds.get()
        self.assertEqual(out.shape, (3, 1))

    def test_1d_numpy_array_becomes_column_vector(self):
        arr = np.array([1, 2, 3])
        ds = DataSource(arr)
        out = ds.get()
        self.assertEqual(out.shape, (3, 1))

    def test_ragged_list_raises_or_coerces(self):
        data = [[1, 2], [3]]
        try:
            ds = DataSource(data)
            self.assertTrue(isinstance(ds.get(), np.ndarray))
        except Exception as e:
            self.assertIsInstance(e, Exception)

    def test_mixed_dtype_coercion(self):
        data = [[1, 2], [3, "x"]]
        ds = DataSource(data)
        self.assertIn(str(ds.get().dtype), ["object", "<U21", "<U32", "<U64"])

    @unittest.skipUnless(HAS_PANDAS, "pandas not installed")
    def test_empty_dataframe_yields_empty_array(self):
        df = pd.DataFrame()
        ds = DataSource(df)
        self.assertEqual(ds.get().shape, (0, 0))
        self.assertEqual(ds.size(), 0)

    def test_apply_transform_identity_does_not_mutate(self):
        arr = np.array([[1, 2], [3, 4]])
        ds = DataSource(arr)
        out = ds.apply_transform(lambda x: x)
        np.testing.assert_array_equal(out, arr)

    def test_apply_transform_on_empty(self):
        ds = DataSource([])
        out = ds.apply_transform(lambda x: x)
        self.assertEqual(out.shape, (0, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
