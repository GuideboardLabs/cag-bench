# Run Summary

- Generated: 2026-05-02 07:40:52
- Source summary: `results/qwen7b_variants_smoke/E_promptonly_K5/summary_plot_ready.csv`
- Task range: 1 to 30
- Modes detected: cag, cag_scoped, cag_oracle_memory, cag_scoped_promptonly
- Trials detected: 3
- Total rows analyzed: 360

> *Continuity terms in this dataset are derived from prior tasks' promoted decisions. CAG memory contains those decisions by construction. Disclose this design when comparing CAG to RAG/DAG.*

Primary comparison uses overall means across all rows (all tasks and all trials), not a single last run.

## Composite score
- Overall mean across all tasks/trials:
  - CAG: 47.06 (vs CAG Scoped +6.57%; vs CAG Oracle +7.19%; vs cag_scoped_promptonly +15.19%).
  - CAG Scoped: 44.16 (vs CAG -6.16%; vs CAG Oracle +0.58%; vs cag_scoped_promptonly +8.10%).
  - CAG Oracle: 43.91 (vs CAG -6.71%; vs CAG Scoped -0.58%; vs cag_scoped_promptonly +7.47%).
  - cag_scoped_promptonly: 40.85 (vs CAG -13.19%; vs CAG Scoped -7.49%; vs CAG Oracle -6.95%).
- Overall ranking by mean: CAG 47.06, CAG Scoped 44.16, CAG Oracle 43.91, cag_scoped_promptonly 40.85.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 0.00, ended at 33.11, change +33.11 points (n/a (start value is 0)), overall increase.
  - CAG Scoped: started at 0.00, ended at 30.21, change +30.21 points (n/a (start value is 0)), overall increase.
  - CAG Oracle: started at 0.00, ended at 32.99, change +32.99 points (n/a (start value is 0)), overall increase.
  - cag_scoped_promptonly: started at 0.00, ended at 28.56, change +28.56 points (n/a (start value is 0)), overall increase.
- Final-task ranking (mean): CAG 33.11, CAG Oracle 32.99, CAG Scoped 30.21, cag_scoped_promptonly 28.56.

## Checklist quality
- Overall mean across all tasks/trials:
  - CAG: 45.69 (vs CAG Scoped +3.33%; vs CAG Oracle +11.23%; vs cag_scoped_promptonly +7.12%).
  - CAG Scoped: 44.22 (vs CAG -3.22%; vs CAG Oracle +7.65%; vs cag_scoped_promptonly +3.67%).
  - CAG Oracle: 41.08 (vs CAG -10.09%; vs CAG Scoped -7.10%; vs cag_scoped_promptonly -3.70%).
  - cag_scoped_promptonly: 42.65 (vs CAG -6.64%; vs CAG Scoped -3.54%; vs CAG Oracle +3.84%).
- Overall ranking by mean: CAG 45.69, CAG Scoped 44.22, cag_scoped_promptonly 42.65, CAG Oracle 41.08.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 100.00, ended at 33.33, change -66.67 points (-66.67%), overall decline.
  - CAG Scoped: started at 95.24, ended at 9.52, change -85.71 points (-90.00%), overall decline.
  - CAG Oracle: started at 90.48, ended at 19.05, change -71.43 points (-78.95%), overall decline.
  - cag_scoped_promptonly: started at 90.48, ended at 28.57, change -61.90 points (-68.42%), overall decline.
- Final-task ranking (mean): CAG 33.33, cag_scoped_promptonly 28.57, CAG Oracle 19.05, CAG Scoped 9.52.

## Source evidence recall
- Overall mean across all tasks/trials:
  - CAG: 40.56 (vs CAG Scoped +8.36%; vs CAG Oracle +4.83%; vs cag_scoped_promptonly +4.19%).
  - CAG Scoped: 37.43 (vs CAG -7.72%; vs CAG Oracle -3.26%; vs cag_scoped_promptonly -3.85%).
  - CAG Oracle: 38.69 (vs CAG -4.61%; vs CAG Scoped +3.36%; vs cag_scoped_promptonly -0.62%).
  - cag_scoped_promptonly: 38.93 (vs CAG -4.02%; vs CAG Scoped +4.01%; vs CAG Oracle +0.62%).
- Overall ranking by mean: CAG 40.56, cag_scoped_promptonly 38.93, CAG Oracle 38.69, CAG Scoped 37.43.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 100.00, ended at 25.00, change -75.00 points (-75.00%), overall decline.
  - CAG Scoped: started at 100.00, ended at 8.33, change -91.67 points (-91.67%), overall decline.
  - CAG Oracle: started at 100.00, ended at 16.67, change -83.33 points (-83.33%), overall decline.
  - cag_scoped_promptonly: started at 100.00, ended at 25.00, change -75.00 points (-75.00%), overall decline.
- Final-task ranking (mean): CAG 25.00, cag_scoped_promptonly 25.00, CAG Oracle 16.67, CAG Scoped 8.33.

## Domain rule recall
- Overall mean across all tasks/trials:
  - CAG: 54.54 (vs CAG Scoped +19.42%; vs CAG Oracle +14.86%; vs cag_scoped_promptonly +13.40%).
  - CAG Scoped: 45.67 (vs CAG -16.26%; vs CAG Oracle -3.82%; vs cag_scoped_promptonly -5.04%).
  - CAG Oracle: 47.48 (vs CAG -12.94%; vs CAG Scoped +3.97%; vs cag_scoped_promptonly -1.27%).
  - cag_scoped_promptonly: 48.09 (vs CAG -11.82%; vs CAG Scoped +5.31%; vs CAG Oracle +1.29%).
- Overall ranking by mean: CAG 54.54, cag_scoped_promptonly 48.09, CAG Oracle 47.48, CAG Scoped 45.67.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 100.00, ended at 50.00, change -50.00 points (-50.00%), overall decline.
  - CAG Scoped: started at 66.67, ended at 50.00, change -16.67 points (-25.00%), overall decline.
  - CAG Oracle: started at 33.33, ended at 50.00, change +16.67 points (+50.00%), overall increase.
  - cag_scoped_promptonly: started at 33.33, ended at 50.00, change +16.67 points (+50.00%), overall increase.
- Final-task ranking (mean): CAG 50.00, CAG Scoped 50.00, CAG Oracle 50.00, cag_scoped_promptonly 50.00.

## Evidence recall
- Overall mean across all tasks/trials:
  - CAG: 47.55 (vs CAG Scoped +14.44%; vs CAG Oracle +10.36%; vs cag_scoped_promptonly +9.28%).
  - CAG Scoped: 41.55 (vs CAG -12.62%; vs CAG Oracle -3.57%; vs cag_scoped_promptonly -4.51%).
  - CAG Oracle: 43.08 (vs CAG -9.39%; vs CAG Scoped +3.70%; vs cag_scoped_promptonly -0.98%).
  - cag_scoped_promptonly: 43.51 (vs CAG -8.49%; vs CAG Scoped +4.72%; vs CAG Oracle +0.99%).
- Overall ranking by mean: CAG 47.55, cag_scoped_promptonly 43.51, CAG Oracle 43.08, CAG Scoped 41.55.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 100.00, ended at 37.50, change -62.50 points (-62.50%), overall decline.
  - CAG Scoped: started at 83.33, ended at 29.17, change -54.17 points (-65.00%), overall decline.
  - CAG Oracle: started at 66.67, ended at 33.33, change -33.33 points (-50.00%), overall decline.
  - cag_scoped_promptonly: started at 66.67, ended at 37.50, change -29.17 points (-43.75%), overall decline.
- Final-task ranking (mean): CAG 37.50, cag_scoped_promptonly 37.50, CAG Oracle 33.33, CAG Scoped 29.17.

## Continuity recall
- Overall mean across all tasks/trials:
  - CAG: 53.26 (vs CAG Scoped +6.64%; vs CAG Oracle +5.20%; vs cag_scoped_promptonly +29.76%).
  - CAG Scoped: 49.94 (vs CAG -6.22%; vs CAG Oracle -1.35%; vs cag_scoped_promptonly +21.69%).
  - CAG Oracle: 50.62 (vs CAG -4.95%; vs CAG Scoped +1.36%; vs cag_scoped_promptonly +23.35%).
  - cag_scoped_promptonly: 41.04 (vs CAG -22.94%; vs CAG Scoped -17.82%; vs CAG Oracle -18.93%).
- Overall ranking by mean: CAG 53.26, CAG Oracle 50.62, CAG Scoped 49.94, cag_scoped_promptonly 41.04.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 85.71, ended at 28.74, change -56.98 points (-66.48%), overall decline.
  - CAG Scoped: started at 80.95, ended at 45.98, change -34.98 points (-43.20%), overall decline.
  - CAG Oracle: started at 85.71, ended at 42.53, change -43.19 points (-50.38%), overall decline.
  - cag_scoped_promptonly: started at 80.95, ended at 20.69, change -60.26 points (-74.44%), overall decline.
- Final-task ranking (mean): CAG Scoped 45.98, CAG Oracle 42.53, CAG 28.74, cag_scoped_promptonly 20.69.

## Token efficiency
- Overall mean across all tasks/trials:
  - CAG: 76.17 (vs CAG Scoped -11.83%; vs CAG Oracle -11.83%; vs cag_scoped_promptonly -11.68%).
  - CAG Scoped: 86.39 (vs CAG +13.42%; vs CAG Oracle +0.00%; vs cag_scoped_promptonly +0.17%).
  - CAG Oracle: 86.39 (vs CAG +13.42%; vs CAG Scoped +0.00%; vs cag_scoped_promptonly +0.17%).
  - cag_scoped_promptonly: 86.25 (vs CAG +13.22%; vs CAG Scoped -0.17%; vs CAG Oracle -0.17%).
- Overall ranking by mean: CAG Scoped 86.39, CAG Oracle 86.39, cag_scoped_promptonly 86.25, CAG 76.17.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 90.02, ended at 62.78, change -27.24 points (-30.26%), overall decline.
  - CAG Scoped: started at 90.02, ended at 84.62, change -5.40 points (-6.00%), overall decline.
  - CAG Oracle: started at 90.02, ended at 84.62, change -5.40 points (-6.00%), overall decline.
  - cag_scoped_promptonly: started at 90.02, ended at 84.60, change -5.42 points (-6.02%), overall decline.
- Final-task ranking (mean): CAG Scoped 84.62, CAG Oracle 84.62, cag_scoped_promptonly 84.60, CAG 62.78.

## Latency efficiency
- Overall mean across all tasks/trials:
  - CAG: 75.41 (vs CAG Scoped -4.28%; vs CAG Oracle -3.34%; vs cag_scoped_promptonly -5.29%).
  - CAG Scoped: 78.78 (vs CAG +4.47%; vs CAG Oracle +0.98%; vs cag_scoped_promptonly -1.05%).
  - CAG Oracle: 78.01 (vs CAG +3.45%; vs CAG Scoped -0.97%; vs cag_scoped_promptonly -2.01%).
  - cag_scoped_promptonly: 79.62 (vs CAG +5.58%; vs CAG Scoped +1.06%; vs CAG Oracle +2.05%).
- Overall ranking by mean: cag_scoped_promptonly 79.62, CAG Scoped 78.78, CAG Oracle 78.01, CAG 75.41.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 80.11, ended at 65.29, change -14.82 points (-18.50%), overall decline.
  - CAG Scoped: started at 83.64, ended at 73.68, change -9.96 points (-11.91%), overall decline.
  - CAG Oracle: started at 83.89, ended at 65.64, change -18.25 points (-21.76%), overall decline.
  - cag_scoped_promptonly: started at 83.54, ended at 72.00, change -11.54 points (-13.81%), overall decline.
- Final-task ranking (mean): CAG Scoped 73.68, cag_scoped_promptonly 72.00, CAG Oracle 65.64, CAG 65.29.

## Memory recall
- Overall mean across all tasks/trials:
  - CAG: 100.00 (vs CAG Scoped +23.93%; vs CAG Oracle +23.93%; vs cag_scoped_promptonly +64.15%).
  - CAG Scoped: 80.69 (vs CAG -19.31%; vs CAG Oracle +0.00%; vs cag_scoped_promptonly +32.46%).
  - CAG Oracle: 80.69 (vs CAG -19.31%; vs CAG Scoped +0.00%; vs cag_scoped_promptonly +32.46%).
  - cag_scoped_promptonly: 60.92 (vs CAG -39.08%; vs CAG Scoped -24.50%; vs CAG Oracle -24.50%).
- Overall ranking by mean: CAG 100.00, CAG Scoped 80.69, CAG Oracle 80.69, cag_scoped_promptonly 60.92.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 100.00, ended at 100.00, change +0.00 points (+0.00%), overall flat trend.
  - CAG Scoped: started at 100.00, ended at 51.72, change -48.28 points (-48.28%), overall decline.
  - CAG Oracle: started at 100.00, ended at 51.72, change -48.28 points (-48.28%), overall decline.
  - cag_scoped_promptonly: started at 100.00, ended at 24.14, change -75.86 points (-75.86%), overall decline.
- Final-task ranking (mean): CAG 100.00, CAG Scoped 51.72, CAG Oracle 51.72, cag_scoped_promptonly 24.14.

## Memory precision
- Overall mean across all tasks/trials:
  - CAG: 39.30 (vs CAG Scoped -25.89%; vs CAG Oracle -25.89%; vs cag_scoped_promptonly -17.67%).
  - CAG Scoped: 53.03 (vs CAG +34.94%; vs CAG Oracle +0.00%; vs cag_scoped_promptonly +11.10%).
  - CAG Oracle: 53.03 (vs CAG +34.94%; vs CAG Scoped +0.00%; vs cag_scoped_promptonly +11.10%).
  - cag_scoped_promptonly: 47.74 (vs CAG +21.46%; vs CAG Scoped -9.99%; vs CAG Oracle -9.99%).
- Overall ranking by mean: CAG Scoped 53.03, CAG Oracle 53.03, cag_scoped_promptonly 47.74, CAG 39.30.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 87.50, ended at 42.65, change -44.85 points (-51.26%), overall decline.
  - CAG Scoped: started at 87.50, ended at 53.57, change -33.93 points (-38.78%), overall decline.
  - CAG Oracle: started at 87.50, ended at 53.57, change -33.93 points (-38.78%), overall decline.
  - cag_scoped_promptonly: started at 87.50, ended at 50.00, change -37.50 points (-42.86%), overall decline.
- Final-task ranking (mean): CAG Scoped 53.57, CAG Oracle 53.57, cag_scoped_promptonly 50.00, CAG 42.65.

## Memory usage rate
- Overall mean across all tasks/trials:
  - CAG: 48.37 (vs CAG Scoped -10.62%; vs CAG Oracle -10.68%; vs cag_scoped_promptonly -10.42%).
  - CAG Scoped: 54.12 (vs CAG +11.88%; vs CAG Oracle -0.08%; vs cag_scoped_promptonly +0.22%).
  - CAG Oracle: 54.16 (vs CAG +11.96%; vs CAG Scoped +0.08%; vs cag_scoped_promptonly +0.30%).
  - cag_scoped_promptonly: 54.00 (vs CAG +11.63%; vs CAG Scoped -0.22%; vs CAG Oracle -0.30%).
- Overall ranking by mean: CAG Oracle 54.16, CAG Scoped 54.12, cag_scoped_promptonly 54.00, CAG 48.37.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 87.50, ended at 34.31, change -53.19 points (-60.78%), overall decline.
  - CAG Scoped: started at 79.17, ended at 91.67, change +12.50 points (+15.79%), overall increase.
  - CAG Oracle: started at 83.33, ended at 83.33, change +0.00 points (+0.00%), overall flat trend.
  - cag_scoped_promptonly: started at 83.33, ended at 76.19, change -7.14 points (-8.57%), overall decline.
- Final-task ranking (mean): CAG Scoped 91.67, CAG Oracle 83.33, cag_scoped_promptonly 76.19, CAG 34.31.

## Continuity per memory token
- Overall mean across all tasks/trials:
  - CAG: 0.28 (vs CAG Scoped -33.87%; vs CAG Oracle -33.87%; vs cag_scoped_promptonly -16.40%).
  - CAG Scoped: 0.42 (vs CAG +51.23%; vs CAG Oracle +0.00%; vs cag_scoped_promptonly +26.43%).
  - CAG Oracle: 0.42 (vs CAG +51.23%; vs CAG Scoped +0.00%; vs cag_scoped_promptonly +26.43%).
  - cag_scoped_promptonly: 0.34 (vs CAG +19.61%; vs CAG Scoped -20.91%; vs CAG Oracle -20.91%).
- Overall ranking by mean: CAG Scoped 0.42, CAG Oracle 0.42, cag_scoped_promptonly 0.34, CAG 0.28.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 1.82, ended at 0.08, change -1.74 points (-95.54%), overall decline.
  - CAG Scoped: started at 1.82, ended at 0.21, change -1.61 points (-88.71%), overall decline.
  - CAG Oracle: started at 1.82, ended at 0.21, change -1.61 points (-88.71%), overall decline.
  - cag_scoped_promptonly: started at 1.82, ended at 0.10, change -1.72 points (-94.75%), overall decline.
- Final-task ranking (mean): CAG Scoped 0.21, CAG Oracle 0.21, cag_scoped_promptonly 0.10, CAG 0.08.

## Contradiction penalty (lower is better)
- Overall mean across all tasks/trials:
  - CAG: 1.50 (vs CAG Scoped +50.00%; vs CAG Oracle +12.50%; vs cag_scoped_promptonly +50.00%).
  - CAG Scoped: 1.00 (vs CAG -33.33%; vs CAG Oracle -25.00%; vs cag_scoped_promptonly +0.00%).
  - CAG Oracle: 1.33 (vs CAG -11.11%; vs CAG Scoped +33.33%; vs cag_scoped_promptonly +33.33%).
  - cag_scoped_promptonly: 1.00 (vs CAG -33.33%; vs CAG Scoped +0.00%; vs CAG Oracle -25.00%).
- Overall ranking by mean: CAG Scoped 1.00, cag_scoped_promptonly 1.00, CAG Oracle 1.33, CAG 1.50.

- Task progression trend (first task mean to last task mean):
  - CAG: started at 0.00, ended at 0.00, change +0.00 points (n/a (start value is 0)), overall flat trend.
  - CAG Scoped: started at 0.00, ended at 0.00, change +0.00 points (n/a (start value is 0)), overall flat trend.
  - CAG Oracle: started at 0.00, ended at 0.00, change +0.00 points (n/a (start value is 0)), overall flat trend.
  - cag_scoped_promptonly: started at 0.00, ended at 0.00, change +0.00 points (n/a (start value is 0)), overall flat trend.
- Final-task ranking (mean): CAG 0.00, CAG Scoped 0.00, CAG Oracle 0.00, cag_scoped_promptonly 0.00.

## Final Boss: Best Rated Task 30 Answer
- Best score: 43.77
- Mode: CAG
- Trial: 1
- Answer:
```text
### Final Implementation Handoff

#### Phased Delivery Roadmap
1. **Phase 1: Core Functionality**
   - Persistence Layer (SQLite)
   - Domain Model (DogProfile, HandlerProfile, TrainingGoal, SessionLog)
   - Assess Lane
2. **Phase 2: Advanced Features**
   - Train Lane
   - Review Lane
   - Import/Export
3. **Phase 3: Safety and Privacy Enhancements**
   - Safety Boundaries (VetReferral, unsafe_to_continue)
   - Privacy Controls (Local passcode, encrypted export)
4. **Phase 4: User Experience and Integration**
   - UX Routes (Dogs, Plan, Session, Progress, Library tabs)
   - Cross-Lane Insights
5. **Phase 5: Testing and Validation**
   - Scenario Fixtures and Deterministic Seeds
   - Cross-Module Integration Coverage
6. **Phase 6: Release and Post-Release Validation**
   - Explicit Checklist and Go/No-Go Criteria

#### Acceptance Gates
1. **Core Functionality Gate**
   - Persistence Layer Testing
   - Domain Model Validation
2. **Advanced Features Gate**
   - Train Lane Implementation
   - Review Lane Integration
3. **Safety and Privacy Gate**
   - Safety Boundary Compliance
   - Privacy Export Verification
4. **User Experience Gate**
   - UX Route Usability Testing
   - Cross-Lane Insights Functionality
5. **Testing Gate**
   - Scenario Fixture Coverage
   - Cross-Module Integration Validation
6. **Release Gate**
   - Explicit Release Checklist Review
   - Post-Release Validation Plan

#### Post-Release Validation
1. **Regression Testing**
   - Triage Protocol for Regression Debugging
   - Conflict Resolver Rules Application
2. **Documentation and Training**
   - ADR-style Architecture Records
   - Operator Runbook
   - Model Boundary Notes
3. **Offline Import/Export**
   - Partial Import Recovery
   - Idempotent Replay
4. **Risk Management**
   - Risk Logging Across Safety and Privacy Boundaries

### Key Files/Modules
1. **Persistence Layer**
   - `persistence.py` (SQLite operations)
2. **Domain Model**
   - `domain_model.py` (DogProfile, HandlerProfile, TrainingGoal, SessionLog)
3. **Assess Lane**
   - `assess_lane.py`
4. **Train Lane**
   - `train_lane.py`
5. **Review Lane**
   - `review_lane.py`
6. **Import/Export**
   - `import_export.py` (JSONL schema versioning, ImportExportManifest)
7. **Safety Boundaries**
   - `safety_boundaries.py` (VetReferral, unsafe_to_continue)
8. **User Experience**
   - `ux_routes.py` (Dogs, Plan, Session, Progress, Library tabs)
9. **Cross-Lane Insights**
   - `cross_lane_insight.py`
10. **Testing**
    - `test_persistence.py`
    - `test_domain_model.py`
    - `test_assess_lane.py`
    - `test_train_lane.py`
    - `test_review_lane.py`
    - `test_import_export.py`
    - `test_safety_boundaries.py`
    - `test_user_experience.py`
    - `test_cross_lane_insight.py`

### Risks and Mitigation
1. **Data Migration**
   - **Risk:** Incompatibility with legacy schema.
   - **Mitigation:** Backfill script, invariant checks, rollback plan.
2. **Safety Compliance**
   - **Risk:** Non-compliance with humane training principles.
   - **Mitigation:** Explicit escalation protocol, safety notes.
3. **Offline Functionality**
   - **Risk:** Partial import recovery issues.
   - **Mitigation:** Idempotent replay, deterministic conflict resolution.

### Testing Strategy
1. **Scenario Fixtures and Seeds**
   - State transitions, privacy tests, import/export tests, safety boundary tests.
2. **Cross-Module Integration Coverage**
   - Assess, Train, Review lanes integrated.
3. **Regression Guardrails**
   - Mapped to prior feature and safety decisions.

This implementation plan aligns with the project's core principles of local-first, humane training, privacy, and repeatable progress tracking. It ensures that each phase is thoroughly tested and validated before moving on to the next, ensuring a robust final product.
```

## Final Boss: Best Fair-Baseline Task 30 Answer
- Best score: 43.77
- Mode: CAG
- Trial: 1
- Answer:
```text
### Final Implementation Handoff

#### Phased Delivery Roadmap
1. **Phase 1: Core Functionality**
   - Persistence Layer (SQLite)
   - Domain Model (DogProfile, HandlerProfile, TrainingGoal, SessionLog)
   - Assess Lane
2. **Phase 2: Advanced Features**
   - Train Lane
   - Review Lane
   - Import/Export
3. **Phase 3: Safety and Privacy Enhancements**
   - Safety Boundaries (VetReferral, unsafe_to_continue)
   - Privacy Controls (Local passcode, encrypted export)
4. **Phase 4: User Experience and Integration**
   - UX Routes (Dogs, Plan, Session, Progress, Library tabs)
   - Cross-Lane Insights
5. **Phase 5: Testing and Validation**
   - Scenario Fixtures and Deterministic Seeds
   - Cross-Module Integration Coverage
6. **Phase 6: Release and Post-Release Validation**
   - Explicit Checklist and Go/No-Go Criteria

#### Acceptance Gates
1. **Core Functionality Gate**
   - Persistence Layer Testing
   - Domain Model Validation
2. **Advanced Features Gate**
   - Train Lane Implementation
   - Review Lane Integration
3. **Safety and Privacy Gate**
   - Safety Boundary Compliance
   - Privacy Export Verification
4. **User Experience Gate**
   - UX Route Usability Testing
   - Cross-Lane Insights Functionality
5. **Testing Gate**
   - Scenario Fixture Coverage
   - Cross-Module Integration Validation
6. **Release Gate**
   - Explicit Release Checklist Review
   - Post-Release Validation Plan

#### Post-Release Validation
1. **Regression Testing**
   - Triage Protocol for Regression Debugging
   - Conflict Resolver Rules Application
2. **Documentation and Training**
   - ADR-style Architecture Records
   - Operator Runbook
   - Model Boundary Notes
3. **Offline Import/Export**
   - Partial Import Recovery
   - Idempotent Replay
4. **Risk Management**
   - Risk Logging Across Safety and Privacy Boundaries

### Key Files/Modules
1. **Persistence Layer**
   - `persistence.py` (SQLite operations)
2. **Domain Model**
   - `domain_model.py` (DogProfile, HandlerProfile, TrainingGoal, SessionLog)
3. **Assess Lane**
   - `assess_lane.py`
4. **Train Lane**
   - `train_lane.py`
5. **Review Lane**
   - `review_lane.py`
6. **Import/Export**
   - `import_export.py` (JSONL schema versioning, ImportExportManifest)
7. **Safety Boundaries**
   - `safety_boundaries.py` (VetReferral, unsafe_to_continue)
8. **User Experience**
   - `ux_routes.py` (Dogs, Plan, Session, Progress, Library tabs)
9. **Cross-Lane Insights**
   - `cross_lane_insight.py`
10. **Testing**
    - `test_persistence.py`
    - `test_domain_model.py`
    - `test_assess_lane.py`
    - `test_train_lane.py`
    - `test_review_lane.py`
    - `test_import_export.py`
    - `test_safety_boundaries.py`
    - `test_user_experience.py`
    - `test_cross_lane_insight.py`

### Risks and Mitigation
1. **Data Migration**
   - **Risk:** Incompatibility with legacy schema.
   - **Mitigation:** Backfill script, invariant checks, rollback plan.
2. **Safety Compliance**
   - **Risk:** Non-compliance with humane training principles.
   - **Mitigation:** Explicit escalation protocol, safety notes.
3. **Offline Functionality**
   - **Risk:** Partial import recovery issues.
   - **Mitigation:** Idempotent replay, deterministic conflict resolution.

### Testing Strategy
1. **Scenario Fixtures and Seeds**
   - State transitions, privacy tests, import/export tests, safety boundary tests.
2. **Cross-Module Integration Coverage**
   - Assess, Train, Review lanes integrated.
3. **Regression Guardrails**
   - Mapped to prior feature and safety decisions.

This implementation plan aligns with the project's core principles of local-first, humane training, privacy, and repeatable progress tracking. It ensures that each phase is thoroughly tested and validated before moving on to the next, ensuring a robust final product.
```

## Final Boss: Task 30 Comparison (Across Runs)
- CAG: score mean 33.11 (n=3), continuity mean 28.74, checklist mean 33.33, source evidence mean 25.00, domain rule mean 50.00, contradiction penalty mean 0.00.
- CAG_SCOPED: score mean 30.21 (n=3), continuity mean 45.98, checklist mean 9.52, source evidence mean 8.33, domain rule mean 50.00, contradiction penalty mean 0.00.
- CAG_ORACLE_MEMORY: score mean 32.99 (n=3), continuity mean 42.53, checklist mean 19.05, source evidence mean 16.67, domain rule mean 50.00, contradiction penalty mean 0.00.
- CAG_SCOPED_PROMPTONLY: score mean 28.56 (n=3), continuity mean 20.69, checklist mean 28.57, source evidence mean 25.00, domain rule mean 50.00, contradiction penalty mean 0.00.

## Phase Summary
- Source: `phase_summary.csv`

### 1-10
- CAG: composite 49.63, continuity 71.20, memory recall 100.00, memory precision 46.55, memory usage rate 62.63, continuity/token 0.63, memory tokens 207.70, prompt tokens 628.70, contradiction penalty 0.00.
- CAG Scoped: composite 49.08, continuity 64.71, memory recall 99.21, memory precision 50.98, memory usage rate 56.78, continuity/token 0.70, memory tokens 158.70, prompt tokens 579.70, contradiction penalty 0.50.
- CAG Oracle: composite 51.11, continuity 72.79, memory recall 99.21, memory precision 50.98, memory usage rate 61.93, continuity/token 0.70, memory tokens 158.70, prompt tokens 579.70, contradiction penalty 0.50.
- cag_scoped_promptonly: composite 48.48, continuity 67.37, memory recall 95.77, memory precision 48.68, memory usage rate 63.15, continuity/token 0.68, memory tokens 162.70, prompt tokens 583.60, contradiction penalty 0.00.

### 11-20
- CAG: composite 58.47, continuity 54.95, memory recall 100.00, memory precision 35.69, memory usage rate 45.79, continuity/token 0.16, memory tokens 674.20, prompt tokens 1064.10, contradiction penalty 1.00.
- CAG Scoped: composite 53.42, continuity 48.87, memory recall 82.63, memory precision 54.35, memory usage rate 48.09, continuity/token 0.35, memory tokens 235.90, prompt tokens 625.80, contradiction penalty 1.00.
- CAG Oracle: composite 53.06, continuity 48.81, memory recall 82.63, memory precision 54.35, memory usage rate 49.06, continuity/token 0.35, memory tokens 235.90, prompt tokens 625.80, contradiction penalty 1.50.
- cag_scoped_promptonly: composite 53.20, continuity 42.82, memory recall 59.93, memory precision 44.18, memory usage rate 53.21, continuity/token 0.25, memory tokens 243.00, prompt tokens 632.70, contradiction penalty 2.00.

### 21-30
- CAG: composite 33.08, continuity 35.42, memory recall 100.00, memory precision 36.39, memory usage rate 38.13, continuity/token 0.09, memory tokens 1144.50, prompt tokens 1523.70, contradiction penalty 3.50.
- CAG Scoped: composite 29.99, continuity 37.72, memory recall 62.09, memory precision 53.57, memory usage rate 57.74, continuity/token 0.25, memory tokens 252.00, prompt tokens 631.20, contradiction penalty 1.50.
- CAG Oracle: composite 27.56, continuity 32.48, memory recall 62.09, memory precision 53.57, memory usage rate 52.26, continuity/token 0.25, memory tokens 252.00, prompt tokens 631.20, contradiction penalty 2.00.
- cag_scoped_promptonly: composite 20.89, continuity 15.57, memory recall 30.54, memory precision 50.45, memory usage rate 46.54, continuity/token 0.12, memory tokens 261.30, prompt tokens 640.40, contradiction penalty 1.00.

### 20-30
- CAG: composite 35.76, continuity 35.91, memory recall 100.00, memory precision 37.56, memory usage rate 37.98, continuity/token 0.09, memory tokens 1122.45, prompt tokens 1489.73, contradiction penalty 3.18.
- CAG Scoped: composite 30.55, continuity 35.56, memory recall 62.60, memory precision 54.86, memory usage rate 54.54, continuity/token 0.25, memory tokens 251.00, prompt tokens 618.27, contradiction penalty 1.36.
- CAG Oracle: composite 27.81, continuity 31.19, memory recall 62.60, memory precision 54.86, memory usage rate 49.47, continuity/token 0.25, memory tokens 251.00, prompt tokens 618.27, contradiction penalty 1.82.
- cag_scoped_promptonly: composite 23.86, continuity 16.01, memory recall 30.99, memory precision 50.62, memory usage rate 46.21, continuity/token 0.12, memory tokens 260.09, prompt tokens 627.27, contradiction penalty 1.82.

### 30 only
- CAG: composite 33.11, continuity 28.74, memory recall 100.00, memory precision 42.65, memory usage rate 34.31, continuity/token 0.08, memory tokens 1234.00, prompt tokens 1675.00, contradiction penalty 0.00.
- CAG Scoped: composite 30.21, continuity 45.98, memory recall 51.72, memory precision 53.57, memory usage rate 91.67, continuity/token 0.21, memory tokens 252.00, prompt tokens 692.00, contradiction penalty 0.00.
- CAG Oracle: composite 32.99, continuity 42.53, memory recall 51.72, memory precision 53.57, memory usage rate 83.33, continuity/token 0.21, memory tokens 252.00, prompt tokens 692.00, contradiction penalty 0.00.
- cag_scoped_promptonly: composite 28.56, continuity 20.69, memory recall 24.14, memory precision 50.00, memory usage rate 76.19, continuity/token 0.10, memory tokens 253.00, prompt tokens 693.00, contradiction penalty 0.00.
