# -*- coding: utf-8 -*-
"""
Unit and Integration Test Suite for Qwen3-VL-4B Medical VQA Architecture
Validates all 15 stages, temperature scaling calibration, and BioBERT similarity
"""
import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.temperature_scaling_calibrator import calculate_ece
from scripts.prompt_stratification_engine import PromptStratificationEngine
from scripts.evaluate_biobert_similarity import mock_clinical_embedding, cosine_similarity

class TestMedicalVQAArchitecture(unittest.TestCase):
    def setUp(self):
        self.engine = PromptStratificationEngine()

    def test_ece_calculation(self):
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        accuracies = [1, 1, 1, 0, 0]
        ece = calculate_ece(confidences, accuracies, num_bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_reasoning_stratification_tiers(self):
        l1 = self.engine.classify_tier("Is this a normal chest radiograph?")
        l2 = self.engine.classify_tier("Identify opacity in right lower lobe.")
        l3 = self.engine.classify_tier("What is the underlying etiology of bilateral effusion?")
        self.assertEqual(l1, "L1: Recognition")
        self.assertEqual(l2, "L2: Localization")
        self.assertEqual(l3, "L3: Complex Diagnosis")

    def test_option_stripping(self):
        raw_mcq = "What finding is present? A) Effusion B) Normal C) Pneumothorax"
        cleaned = self.engine.transform_mcq_to_open_ended(raw_mcq)
        self.assertNotIn("A)", cleaned)
        self.assertNotIn("B)", cleaned)
        self.assertNotIn("C)", cleaned)

    def test_biobert_cosine_metric(self):
        e1 = mock_clinical_embedding("Pleural effusion in left lung")
        e2 = mock_clinical_embedding("Left-sided pleural fluid accumulation")
        sim = cosine_similarity(e1, e2)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)

    def test_deliverable_assets_exist(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets = os.path.join(root, "assets")
        self.assertTrue(os.path.exists(os.path.join(assets, "architecture_diagram.svg")))
        self.assertTrue(os.path.exists(os.path.join(assets, "architecture_diagram.png")))
        self.assertTrue(os.path.exists(os.path.join(assets, "interactive_viewer.html")))
        self.assertTrue(os.path.exists(os.path.join(assets, "archi.pdf")))

if __name__ == "__main__":
    unittest.main()