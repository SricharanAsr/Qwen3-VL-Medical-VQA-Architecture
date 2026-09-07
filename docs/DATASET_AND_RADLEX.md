# PMC-VQA Dataset Curation & RadLex Ontology Canonicalization

## 1. Single-Organ Filtering Protocol (Stage 1)
The PMC-VQA dataset comprises multimodal biomedical figures across varied anatomical systems (neuro, abdominal, musculoskeletal, and thoracic). To isolate pulmonary pathology:

- **Modality Filter:** Radiographs only (`Modality == "Chest X-Ray"` or `"Radiograph"`).
- **Anatomical Target:** Thoracic cavity, pulmonary parenchyma, mediastinum, pleura, and rib cage.
- **Projection Standards:** Posteroanterior (PA) and Anteroposterior (AP) views. Lateral and oblique projections are categorized under separate sub-cohorts.
- **Exclusion Criteria:** CT slices, MRI sequences, ultrasound scans, histopathological micrographs, and pediatric radiographs with confounding developmental variants.

---

## 2. RadLex Standardization Index (Stage 2)

| Colloquial / Unstandardized Phrase | Canonical RadLex Term | RadLex RID | Anatomical Region |
|---|---|---|---|
| "spot on right lung" | Pulmonary nodule, right lung | `RID:28491` | Right Lung |
| "water in the lungs" | Pleural effusion | `RID:4832` | Pleural Space |
| "collapsed lung" | Pneumothorax | `RID:4839` | Pleural Cavity |
| "enlarged heart" | Cardiomegaly | `RID:3314` | Cardiac Silhouette |
| "patchy lung cloudiness" | Consolidation / Infiltrate | `RID:4828` | Alveolar Space |
| "broken rib" | Rib fracture | `RID:49814` | Thoracic Cage |
| "air in mediastinum" | Pneumomediastinum | `RID:5213` | Mediastinum |
| "scar tissue on apex" | Apical pleural thickening | `RID:34521` | Apical Pleura |

---

## 3. Canonicalization Normalization Pipeline
```python
def canonicalize_finding(raw_text: str) -> str:
    """Replaces non-standard clinical synonyms with official RSNA RadLex entities."""
    for colloquial, (canonical, rid) in RADLEX_MAP.items():
        if colloquial in raw_text.lower():
            raw_text = raw_text.replace(colloquial, f"{canonical} [{rid}]")
    return raw_text
```