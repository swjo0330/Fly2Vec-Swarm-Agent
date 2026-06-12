# Fly2Vec RAG Search E2E Verification Report

- **Date**: 2026-05-04
- **Qdrant**: localhost:6333
- **Collection**: `fly2vec_mission_paths`
- **Python**: `~/.pyenv/versions/3.12.13/bin/python3`

---

## 1. Qdrant API Direct Verification

### 1-1. Collection List
```
GET /collections → status: ok
Collections: ["fly2vec_mission_paths"]
```
**Result**: PASS

### 1-2. Collection Detail
| Item | Value |
|------|-------|
| points_count | 2,261 |
| vector_dim | 128 |
| distance | Cosine |
| status | green |
| indexed_vectors | 0 (HNSW threshold=10,000, brute-force scan) |
| optimizer_status | ok |

**Result**: PASS

### 1-3. Scroll Sample (3 points)

| # | ID | mission_id | mission_type | wp_count | wp_spot_id |
|---|-------|------------|--------------|----------|------------|
| 1 | 001d3439... | MP2023112204 | grid_search | 6 | 231122-CITY-EC2023112206 |
| 2 | 0037900f... | MP2021040706 | grid_search | 26 | 210407-songdo_D |
| 3 | 005295ba... | MP2024070213 | linear_flight | 8 | 240702-테스트2222 |

Payload schema: `{mission_id, mission_type, wp_count, wp_spot_id}`

**Result**: PASS

---

## 2. find_similar Verification

Query: stored vector of MP2023112204 (grid_search, 6 WP)

| Rank | Score | mission_id | mission_type | wp_count | wp_spot_id |
|------|-------|------------|--------------|----------|------------|
| 1 | 1.000000 | MP2023112204 | grid_search | 6 | 231122-CITY-EC2023112206 |
| 2 | 0.997757 | MP2023111001 | grid_search | 6 | 231110-소양강가로 시연_01 |
| 3 | 0.993927 | MP2021030401 | grid_search | 6 | 210304-PhanTest_2 |
| 4 | 0.990567 | MP2023103001 | grid_search | 6 | 231030-소양세로01 |
| 5 | 0.990022 | MP2021081203 | waypoint_sequence | 6 | 210812-T_T_03 |

- Search time: **22.0ms**
- Self-match returns score=1.0 (correct)
- Top-5 all have same wp_count=6, mostly grid_search type (semantic coherence confirmed)

**Result**: PASS

---

## 3. Anomaly Score Verification

### Normal Mission Vector (MP2023112204)
| Component | Score |
|-----------|-------|
| centroid_distance | 0.0093 |
| isolation_forest | 0.0139 |
| knn_average | 0.0093 |
| **final_score** | **0.0112** |
| is_anomalous | **False** |

### Random Vector (np.random, seed=42)
| Component | Score |
|-----------|-------|
| centroid_distance | 0.7900 |
| isolation_forest | 1.0000 |
| knn_average | 0.7900 |
| **final_score** | **0.8740** |
| is_anomalous | **True** |

- Normal vs Random separation: 0.0112 vs 0.8740 (78x difference)
- Threshold (0.5) correctly classifies both cases

**Result**: PASS

---

## 4. Node2Vec Embedding to Qdrant Search E2E

### Input PG JSON
```json
{
  "waypoints": [
    {"lat_deg": 37.55, "lon_deg": 126.97, "alt_m": 50, "seq": 0},
    {"lat_deg": 37.56, "lon_deg": 126.98, "alt_m": 50, "seq": 1},
    {"lat_deg": 37.57, "lon_deg": 126.97, "alt_m": 50, "seq": 2},
    {"lat_deg": 37.56, "lon_deg": 126.96, "alt_m": 50, "seq": 3}
  ]
}
```

### Encoding Result
- Dimension: 128
- Vector[:5]: [-0.042756, 0.014965, 0.152866, 0.048873, -0.071183]
- Norm: 1.0514
- Encode time: **695.7ms**

### Search Result (top-5)
| Rank | Score | mission_id | mission_type | wp_count |
|------|-------|------------|--------------|----------|
| 1 | 0.999974 | MP2020121101 | waypoint_sequence | 4 |
| 2 | 0.999734 | MP2024011605 | grid_search | 4 |
| 3 | 0.999694 | MP2023071115 | grid_search | 4 |
| 4 | 0.999657 | MP2023071105 | grid_search | 4 |
| 5 | 0.999637 | MP2024022107 | grid_search | 4 |

- All top-5 have wp_count=4 (matches input WP count)
- Search time: **89.9ms**
- Anomaly score: **0.0006** (not anomalous)

### E2E Timing
| Phase | Time |
|-------|------|
| Node2Vec encode | 695.7ms |
| Qdrant search | 89.9ms |
| **Total** | **785.6ms** |

**Result**: PASS

---

## 5. Summary

| Test | Status | Notes |
|------|--------|-------|
| Qdrant API (collections) | PASS | 2,261 points, 128d, Cosine |
| Qdrant API (scroll) | PASS | Payload: mission_id/type/wp_count/spot_id |
| find_similar | PASS | 22ms, semantically coherent results |
| anomaly_score (normal) | PASS | 0.0112, correctly classified as normal |
| anomaly_score (random) | PASS | 0.8740, correctly classified as anomalous |
| E2E (encode+search) | PASS | 786ms total, wp_count matching |

### Observations
1. **indexed_vectors_count=0**: HNSW index not built because point count (2,261) < indexing_threshold (10,000). All searches use brute-force scan. Performance is still excellent (<100ms) at this scale.
2. **Node2Vec stochasticity**: Each `encode_mission()` call trains a fresh Node2Vec model, so the same PG JSON produces slightly different vectors across calls. For production, consider caching or pre-training a model.
3. **Anomaly detection**: The 3-method ensemble effectively separates normal (0.01) from anomalous (0.87) vectors. The iso_forest component is approximated (distance-based), not a true Isolation Forest.
4. **Semantic coherence**: find_similar correctly groups missions by structural similarity (wp_count, mission_type).

### Issues Found
- **None blocking**. All 6 tests passed successfully.
